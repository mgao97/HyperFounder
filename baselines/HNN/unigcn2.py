import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch_scatter
from .utils import normalize_l2


class UniGCNIIConv(nn.Module):
    def __init__(self, args, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.args = args
        self.reduce = args.reduce

    def reset_parameters(self):
        self.W.reset_parameters()
        
    def forward(self, X, vertex, edges, alpha, beta, X0):
        N = X.shape[-2]

        Xve = X[..., vertex, :]  # [nnz, C]
        Xe = torch_scatter.scatter(Xve, edges, dim=-2, reduce=self.reduce)  # [E, C]

        Xev = Xe[..., edges, :]  # [nnz, C]
        Xv = torch_scatter.scatter(Xev, vertex, dim=-2, reduce=self.reduce, dim_size=N)  # [N, C]

        X = Xv 

        if self.args.use_norm:
            X = normalize_l2(X)

        Xi = (1 - alpha) * X + alpha * X0
        X = (1 - beta) * Xi + beta * self.W(Xi)

        return X, Xe

    def flops(self, X, vertex, edges):
        flops = vertex.shape[0] + edges.shape[0] + edges.shape[0] + vertex.shape[0]
        flops += np.prod(X.shape)
        flops += np.prod(X.shape[:-1]) * self.in_features * self.out_features
        return flops


class UniGCNII(nn.Module):
    def __init__(self, num_features, num_targets, args):

        super().__init__()
        self.num_layers = args.All_num_layers
        self.nhid = args.MLP_hidden

        self.in_features = num_features
        self.out_features = num_targets

        act = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)
        self.alpha = args.restart_alpha
        self.lamda = args.lamda

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(num_features, self.nhid))
        for _ in range(self.num_layers):
            self.convs.append(UniGCNIIConv(args, self.nhid, self.nhid))
        self.convs.append(torch.nn.Linear(self.nhid, num_targets))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters()) + list(self.convs[-1:].parameters())

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
    def forward(self, data):
        x = data.x
        V, E = data.hyperedge_index[0], data.hyperedge_index[1]
        lamda, alpha = self.lamda, self.alpha
        x = self.dropout(x)
        x = F.relu(self.convs[0](x)) 
        x0 = x 
        for i, con in enumerate(self.convs[1:-1]):
            x = self.dropout(x)
            beta = math.log(lamda / (i + 1) + 1)
            x, e = con(x, V, E, alpha, beta, x0)
            x = F.relu(x)
        x = self.dropout(x)
        x = self.convs[-1](x)
        return x, e

    def flops(self, data):
        x = data.x
        V, E = data.edge_index[0], data.edge_index[1]
        flops = self.in_features * self.nhid * np.prod(x.shape[:-1])
        flops += self.nhid * np.prod(x.shape[:-1])
        for i, con in enumerate(self.convs[1:-1]):
            flops += con.flops(x, V, E)
            flops += self.nhid * np.prod(x.shape[:-1])
        flops = self.out_features * self.nhid * np.prod(x.shape[:-1])
        return flops
