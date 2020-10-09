import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils.util import convert_to_gpu

# (64, 64, 207, 1)


class STGCNModel(nn.Module):
    def __init__(self, supports, args):
        super(STGCNModel, self).__init__()
        t_len = args['seq_output'] - 2 * (args['Kt'] - 1)
        self.st_layer1 = STLayer(supports, args['input_size'], args['tcn_size'], t_len, args)
        t_len -= 2 * (args['Kt'] - 1)
        self.st_layer2 = STLayer(supports, args['tcn_size'], args['tcn_size'], t_len, args)
        self.last_temporal_layer = TemporalLayer(args['tcn_size'], args['tcn_size'], t_len)
        self.output_layer = nn.Conv1d(args['tcn_size'], args['seq_output'], 1)
        # self.model_seq = nn.Sequential(st_layer1, st_layer2, last_temporal_layer, output_layer)
        self.seq_input = args['seq_input']

    def forward(self, inputs, **kwargs):
        inputs = inputs[:, :self.seq_input, :, :]
        inputs = torch.transpose(inputs, dim0=1, dim1=3)
        # outputs = self.model_seq(inputs)
        outputs = self.st_layer1(inputs)
        outputs = self.st_layer2(outputs)
        outputs = self.last_temporal_layer(outputs)
        outputs = self.output_layer(outputs.squeeze())
        return outputs


class STLayer(nn.Module):
    def __init__(self, supports, input_size, output_size, t_output_len, args):
        super(STLayer, self).__init__()
        self.temp_layer1 = TemporalLayer(input_size, args['tcn_size'], args['Kt'])
        self.graphconv_layer = GraphConv(args['tcn_size'], args['gcn_size'], supports)
        self.temp_layer2 = TemporalLayer(args['gcn_size'], args['tcn_size'], args['Kt'])
        self.layer_norm = nn.LayerNorm([args['tcn_size'], args['num_nodes'], t_output_len])
        self.dropout = args['dropout']

    def forward(self, inputs):
        inputs = self.temp_layer1(inputs)
        inputs = F.relu(self.graphconv_layer(inputs))
        inputs = self.temp_layer2(inputs)
        inputs = self.layer_norm(inputs)
        inputs = F.dropout(inputs, self.dropout, training=self.training)
        return inputs


class TemporalLayer(nn.Module):
    def __init__(self, input_size, output_size, Kt):
        super(TemporalLayer, self).__init__()
        self.tcn1 = nn.Conv2d(input_size, output_size, (1, Kt))
        self.tcn2 = nn.Conv2d(input_size, output_size, (1, Kt))
        self.Kt = Kt
        self.output_size = output_size
        self.input_size = input_size

    def forward(self, inputs):
        inputs_res = F.pad(inputs[:, :, :, self.Kt - 1:], (0, 0, 0, 0, 0, self.output_size - self.input_size))
        outputs = (self.tcn1(inputs) + inputs_res) * F.sigmoid(self.tcn2(inputs))
        return outputs


class GraphConv(nn.Module):
    def __init__(self, input_size, output_size, supports):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(size=(supports.shape[0], input_size, output_size)))
        self.bias = nn.Parameter(torch.FloatTensor(size=(output_size,)))
        self.supports = supports
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.bias.data, val=0.0)

    def forward(self, inputs):
        aggr = torch.einsum('khw,bfwt->bkfht', self.supports, inputs)
        output = torch.einsum('bkfht,kfo->boht', aggr, self.weight)
        output = torch.transpose(torch.transpose(output, dim0=1, dim1=3) + self.bias, dim0=1, dim1=3)
        return output
