import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter
from baselines.HNN.utils import zeros,glorot,normalize_l2

# v1: X -> XW -> AXW -> norm
class UniGINConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=1, dropout=0., negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.eps = nn.Parameter(torch.Tensor([0.]))
        self.args = args 

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        zeros(self.eps)
    
    def forward(self, X, vertex, edges):
        N = X.shape[0]
        
        # v1: X -> XW -> AXW -> norm
        X = self.W(X) 

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, C]
        
        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        X = (1 + self.eps) * X + Xv 

        if self.args.use_norm:
            X = normalize_l2(X)

        return X,Xe

__all_convs__ = {
'UniGIN': UniGINConv,
}

class UniGNN(nn.Module):
    def __init__(self,num_features, num_targets, args):

        super(UniGNN, self).__init__()
        UniGNNConv = __all_convs__[args.method]
        self.num_layers = args.All_num_layers
        self.nhead=args.uni_heads
        self.hidden_dim = args.MLP_hidden
        
        self.conv_out = UniGNNConv(args, self.hidden_dim * self.nhead, num_targets, heads=1, dropout=args.attn_drop)
        self.convs = nn.ModuleList(
            [UniGNNConv(args, num_features, self.hidden_dim, heads=self.nhead, dropout=args.attn_drop)] +
            [UniGNNConv(args, self.hidden_dim * self.nhead, self.hidden_dim, heads=self.nhead, dropout=args.attn_drop) for _ in range(self.num_layers-2)]
        )
        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, data):
        
        X=data.x
        V,E=data.hyperedge_index[0],data.hyperedge_index[1]
        
        X = self.input_drop(X)
        for conv in self.convs:
            X,_ = conv(X, V, E)
            X = self.act(X)
            X = self.dropout(X)

        X,Xe = self.conv_out(X, V, E)      
        return X,Xe