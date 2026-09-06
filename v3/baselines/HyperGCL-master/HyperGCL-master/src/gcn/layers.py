# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# from gcn.layers import GraphConvolution, MLPLayer


# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         return x

# class MLP(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(MLP, self).__init__()

#         self.layer1 = MLPLayer(nfeat, nhid)
#         self.layer2 = MLPLayer(nhid, nclass)
#         self.dropout = dropout
        
#     def forward(self, x):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.relu(self.layer1(x))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.layer2(x)
#         return x

# class LAGCN(nn.Module):
#     def __init__(self, concat, nfeat, nhid, nclass, dropout):
#         super(LAGCN, self).__init__()

#         self.gcn1_list = nn.ModuleList()
#         for _ in range(concat):
#             self.gcn1_list.append(GraphConvolution(nfeat, nhid))
#         self.gc2 = GraphConvolution(concat*nhid, nclass)
#         self.dropout = dropout

#     def forward(self, x_list, adj):
#         hidden_list = []
#         for k, con in enumerate(self.gcn1_list):
#             x = F.dropout(x_list[k], self.dropout, training=self.training)
#             hidden_list.append(F.relu(con(x, adj)))
#         x = torch.cat((hidden_list), dim=-1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         return x

import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class MLPLayer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
