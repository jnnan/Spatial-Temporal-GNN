import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import convert_to_gpu, create_diffusion_supports_dense


class GraphWavenet(nn.Module):
    def __init__(self, supports, args):
        super(GraphWavenet, self).__init__()
        self.seq_input = args['seq_input']
        self.node_vec1 = convert_to_gpu(nn.Parameter((torch.randn(args['num_nodes'], 10)), requires_grad=True))
        self.node_vec2 = convert_to_gpu(nn.Parameter((torch.randn(10, args['num_nodes'])), requires_grad=True))
        self.supports = supports
        self.num_matrices = (args['max_diffusion_step'] + 1) * 3
        self.max_diffusion_step = args['max_diffusion_step']
        self.num_nodes = args['num_nodes']
        STLayers = []
        for i in range(4):
            for j in range(1, 3):
                STLayers.append(STLayer(args['hidden_size'], args['hidden_size'], args['skip_size'], j, self.num_matrices))
        self.STLayers = nn.ModuleList(STLayers)
        self.init_linear = nn.Conv2d(args['input_size'], args['hidden_size'], (1, 1))
        self.end_conv_1 = nn.Conv2d(args['skip_size'], args['end_size'], kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(args['end_size'], args['seq_output'], kernel_size=(1, 1), bias=True)

    def forward(self, inputs, **kwargs):
        adaptive_adj = F.softmax(F.relu(torch.mm(self.node_vec1, self.node_vec2)), dim=1)
        adaptive_supports = create_diffusion_supports_dense([adaptive_adj], self.max_diffusion_step, self.num_nodes)
        supports = torch.cat([self.supports, adaptive_supports], dim=0)
        inputs = inputs[:, :self.seq_input, :, :]
        inputs = torch.transpose(inputs, dim0=1, dim1=3)
        inputs = F.pad(inputs, (1, 0))
        inputs = self.init_linear(inputs)
        skip_sum = 0
        for stlayer in self.STLayers:
            inputs, skip_out = stlayer(inputs, supports)
            skip_sum = skip_sum + skip_out
        skip_sum = torch.unsqueeze(skip_sum, dim=3)
        outputs = self.end_conv_2(F.relu(self.end_conv_1(F.relu(skip_sum))))
        return torch.unsqueeze(outputs, dim=3)


class STLayer(nn.Module):
    def __init__(self, input_size, output_size, skip_size, dilation, num_matrices):
        super(STLayer, self).__init__()
        self.tcn1 = nn.Conv2d(input_size, output_size, (1, 2), dilation=dilation)
        self.tcn2 = nn.Conv2d(input_size, output_size, (1, 2), dilation=dilation)
        self.gcn = GraphConv(output_size, output_size, num_matrices)
        self.skip_linear = nn.Conv1d(output_size, skip_size, 1)
        self.bn = nn.BatchNorm2d(output_size)

    def forward(self, inputs, supports):
        gate_out = torch.tanh(self.tcn1(inputs)) * torch.sigmoid(self.tcn2(inputs))
        skip_out = self.skip_linear(gate_out[:, :, :, -1])
        gate_out = self.bn(self.gcn(gate_out, supports) + inputs[:, :, :, -gate_out.shape[3]:])
        return gate_out, skip_out


class GraphConv(nn.Module):
    def __init__(self, input_size, output_size, num_matrices):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(size=(num_matrices, input_size, output_size)))
        self.bias = nn.Parameter(torch.FloatTensor(size=(output_size,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.bias.data, val=0.0)

        # inputs(64, 2, 207, 12)

    def forward(self, inputs, supports):
        aggr = torch.einsum('khw,bfwt->bkfht', supports, inputs)
        output = torch.einsum('bkfht,kfo->boht', aggr, self.weight)
        output = torch.transpose(torch.transpose(output, dim0=1, dim1=3) + self.bias, dim0=1, dim1=3)
        return output
