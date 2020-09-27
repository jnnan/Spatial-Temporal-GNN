import torch
import torch.nn as nn
import random
import math
from utils.util import convert_to_gpu


class DCRNNmodel(nn.Module):
    def __init__(self, supports, args):
        super(DCRNNmodel, self).__init__()
        self.seq_input = args['seq_input']
        self.seq_output = args['seq_output']
        self.cl_decay = args['cl_decay']
        self.encoder = DCRNNEncoder(supports, args)
        self.decoder = DCRNNDecoder(supports, args)

    def forward(self, input, **kwargs):
        source = input[:, :self.seq_input, :, :]
        target = input[:, self.seq_input:self.seq_input+self.seq_output, :, :]
        go_symbol = convert_to_gpu(torch.zeros_like(input[:, 0, :, :]))
        go_symbol = torch.unsqueeze(go_symbol, dim=1)
        target = torch.cat([go_symbol, target], dim=1)
        target = target[..., 0]
        target = torch.unsqueeze(target, dim=3)
        context = self.encoder(source)
        if kwargs['is_eval']:
            outputs = self.decoder(target, context, teacher_force_ratio=0)
        else:
            outputs = self.decoder(target, context, teacher_force_ratio=self.compute_ratio(kwargs['global_step'], self.cl_decay))
        return outputs

    @staticmethod
    def compute_ratio(global_step, k):
        return k / (k + math.exp(global_step / k))


class DCRNNEncoder(nn.Module):
    def __init__(self, supports, args):
        super(DCRNNEncoder, self).__init__()
        batch_size = args['batch_size']
        num_nodes = args['num_nodes']
        input_size = args['input_size']
        hidden_size = args['hidden_size']
        num_layers = args['num_layers']
        encoding_cells = []
        encoding_cells.append(DCGRUCell(supports, input_size, hidden_size))
        for _ in range(1, num_layers):
            encoding_cells.append(DCGRUCell(supports, hidden_size, hidden_size))
        self.encoding_cells = nn.ModuleList(encoding_cells)
        self.init_hidden = convert_to_gpu(torch.zeros(batch_size, num_nodes, hidden_size))

    def forward(self, input):
        seq_length = input.shape[1]

        current_input = input
        output_hidden = []
        for i_layer in range(len(self.encoding_cells)):
            hidden_state = self.init_hidden
            output_inner = []
            for t in range(seq_length):
                _, hidden_state = self.encoding_cells[i_layer](current_input[:, t, :, :], hidden_state)
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)
            current_input = torch.stack(output_inner, dim=1)
        return output_hidden


class DCRNNDecoder(nn.Module):
    def __init__(self, supports, args):
        super(DCRNNDecoder, self).__init__()
        input_size = 1
        hidden_size = args['hidden_size']
        num_layers = args['num_layers']
        decoding_cells = []
        decoding_cells.append(DCGRUCell(supports, input_size, hidden_size))
        for _ in range(1, num_layers - 1):
            decoding_cells.append(DCGRUCell(supports, hidden_size, hidden_size))
        decoding_cells.append(DCGRUCell(supports, hidden_size, hidden_size, True))
        self.decoding_cells = nn.ModuleList(decoding_cells)

    def forward(self, input, init_hidden, teacher_force_ratio):
        seq_length = input.shape[1]
        current_input = input[:, 0, :, :]
        outputs = []
        for t in range(1, seq_length):
            next_input_hidden_state = []
            for i_layer in range(0, len(self.decoding_cells)):
                hidden_state = init_hidden[i_layer]
                output, hidden_state = self.decoding_cells[i_layer](current_input, hidden_state)
                current_input = output
                next_input_hidden_state.append(hidden_state)
            init_hidden = torch.stack(next_input_hidden_state, dim=0)
            outputs.append(output)
            teacher_force = random.random() < teacher_force_ratio
            current_input = (input[:, t, :, :] if teacher_force else output)
        outputs = torch.stack(outputs, dim=1)
        return outputs


class DCGRUCell(nn.Module):
    def __init__(self, supports, input_size, output_size, need_proj=False):
        super(DCGRUCell, self).__init__()
        self.graph_conv_r = GraphConv(supports, input_size+output_size, output_size)
        self.graph_conv_c = GraphConv(supports, input_size+output_size, output_size)
        self.graph_conv_u = GraphConv(supports, input_size+output_size, output_size)
        self.need_proj = need_proj
        if need_proj:
            self.proj = nn.Linear(output_size, 1)

    def forward(self, input, state):
        r = torch.sigmoid(self.graph_conv_r(torch.cat([input, state], dim=2)))
        u = torch.sigmoid(self.graph_conv_u(torch.cat([input, state], dim=2)))
        c = torch.tanh(self.graph_conv_c(torch.cat([input, r * state], dim=2)))
        output = state_new = u * state + (1 - u) * c
        if self.need_proj:
            output = self.proj(state_new)
        return output, state_new


class GraphConv(nn.Module):
    def __init__(self, supports, input_size, output_size):
        super(GraphConv, self).__init__()
        self.supports = supports
        num_matrices = supports.shape[0]
        self.weight = nn.Parameter(torch.FloatTensor(size=(num_matrices, input_size, output_size)))
        self.bias = nn.Parameter(torch.FloatTensor(size=(output_size,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.bias.data, val=0.0)

    # aggr(64, 6, 207, 66)   
    # weight(6, 66, 64)

    def forward(self, inputs):
        aggr = torch.einsum('jkm,iml->ijkl', self.supports, inputs)
        output = torch.einsum('ijkl,jlm->ikm', aggr, self.weight)
        output = output + self.bias
        return output

'''
inputs = torch.randn(64, 207, 66)
supports = torch.randn(6, 207, 207)
weight = torch.randn(6, 66, 64)
def forward(inputs, supports, weight):
        aggr = torch.einsum('jkm,iml->ijkl', supports, inputs)
        print(aggr.shape)
        print(weight.shape)
        output = torch.einsum('ijkl,jlm->ikm', aggr, weight)
        return output
output = forward(inputs, supports, weight)
print(output.shape)
'''