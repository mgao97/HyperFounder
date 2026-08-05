import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import SparseMM


class HJRLConv(nn.Module):

    def __init__(self, in_features, out_features, activation='leaky_relu', neg_slope=0, bias=False):
        super(HJRLConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        act = {'relu': F.relu, 'elu': F.elu, 'leaky_relu': F.leaky_relu}
        if self.activation is not None:
            self.act = act[activation]

        self.neg_slope = neg_slope
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_features, adj):
        support = SparseMM.apply(input_features, self.weight)
        output = SparseMM.apply(adj, support)

        if self.bias is not None:
            output = output + self.bias
        elif self.activation == None:
            output = output
        elif self.activation == 'leaky_relu':
            output = self.act(output, negative_slope=self.neg_slope)
        else:
            output = self.act(output)
        
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class HJRL(nn.Module):
    def __init__(self, num_features, num_targets, args):
        super(HJRL, self).__init__()

        self.num_layers = args.All_num_layers
        self.hidden_dim = args.MLP_hidden
        self.dropout = args.dropout
        self.activition = args.activation
        self.neg_slope = args.neg_slope

        self.convs = nn.ModuleList()
        if self.num_layers == 1:
            self.convs.append(HJRLConv(num_features, num_targets, activation=None))
        else:
            self.convs.append(HJRLConv(num_features, self.hidden_dim, self.activition, self.neg_slope))
            for _ in range(self.num_layers - 2):
                self.convs.append(HJRLConv(self.hidden_dim, self.hidden_dim, self.activition, self.neg_slope))
            self.convs.append(HJRLConv(self.hidden_dim, num_targets, activation=None))

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()

    def forward(self, data):
        x, y, HHT, H, HT, HTH = data.norm_X, data.norm_E, data.HHT, data.H, data.HT, data.HTH
        
        if self.num_layers == 1:
            HHT_l = self.convs[0](x, HHT)
            H_l = self.convs[0](y, H)
            HT_l = self.convs[0](x, HT)
            HTH_l = self.convs[0](y, HTH)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                HHT_l = self.convs[i](x, HHT)
                H_l = self.convs[i](y, H)
                HT_l = self.convs[i](x, HT)
                HTH_l = self.convs[i](y, HTH)

                HHT_l = F.dropout(HHT_l, self.dropout, training=self.training)
                H_l = F.dropout(H_l, self.dropout, training=self.training)
                HT_l = F.dropout(HT_l, self.dropout, training=self.training)
                HTH_l = F.dropout(HTH_l, self.dropout, training=self.training)

                x = HHT_l + H_l
                y = HT_l + HTH_l

            HHT_l = self.convs[-1](x, HHT)
            H_l = self.convs[-1](y, H)
            HT_l = self.convs[-1](x, HT)
            HTH_l = self.convs[-1](y, HTH)

        HHT_l = F.leaky_relu(HHT_l, negative_slope=self.neg_slope)
        H_l = F.leaky_relu(H_l, negative_slope=self.neg_slope)
        HT_l = F.leaky_relu(HT_l, negative_slope=self.neg_slope)
        HTH_l = F.leaky_relu(HTH_l, negative_slope=self.neg_slope)

        x = HHT_l + H_l
        y = HT_l + HTH_l

        return x, y
