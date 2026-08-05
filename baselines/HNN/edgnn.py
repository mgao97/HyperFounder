import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLP
import torch_scatter


class EquivSetConv(nn.Module):
    def __init__(self, in_features, out_features, mlp1_layers=1, mlp2_layers=1,
                 mlp3_layers=1, aggr='add', alpha=0.5, dropout=0., normalization='None', input_norm=False):
        super().__init__()

        if mlp1_layers > 0:
            self.W1 = MLP(in_features, out_features, out_features, mlp1_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W1 = nn.Identity()

        if mlp2_layers > 0:
            self.W2 = MLP(in_features + out_features, out_features, out_features, mlp2_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W2 = lambda X: X[..., in_features:]

        if mlp3_layers > 0:
            self.W = MLP(out_features, out_features, out_features, mlp3_layers,
                dropout=dropout, Normalization=normalization, InputNorm=input_norm)
        else:
            self.W = nn.Identity()
        self.aggr = aggr
        self.alpha = alpha
        self.dropout = dropout

    def reset_parameters(self):
        if isinstance(self.W1, MLP):
            self.W1.reset_parameters()
        if isinstance(self.W2, MLP):
            self.W2.reset_parameters()
        if isinstance(self.W, MLP):
            self.W.reset_parameters()

    def forward(self, X, vertex, edges, X0):
        N = X.shape[-2]

        Xve = self.W1(X)[..., vertex, :]  # [nnz, C]
        Xe = torch_scatter.scatter(Xve, edges, dim=-2, reduce=self.aggr)  # [E, C]
        
        Xev = Xe[..., edges, :]  # [nnz, C]
        Xev = self.W2(torch.cat([X[..., vertex, :], Xev], -1))
        Xv = torch_scatter.scatter(Xev, vertex, dim=-2, reduce=self.aggr, dim_size=N)  # [N, C]

        X = Xv

        X = (1 - self.alpha) * X + self.alpha * X0
        X = self.W(X)

        return X, Xe


class JumpLinkConv(nn.Module):
    def __init__(self, in_features, out_features, mlp_layers=2, aggr='add', alpha=0.5):
        super().__init__()
        self.W = MLP(in_features, out_features, out_features, mlp_layers,
            dropout=0., Normalization='None', InputNorm=False)

        self.aggr = aggr
        self.alpha = alpha

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, X, vertex, edges, X0, beta=1.):
        N = X.shape[-2]

        Xve = X[..., vertex, :]  # [nnz, C]
        Xe = torch_scatter.scatter(Xve, edges, dim=-2, reduce=self.aggr)  # [E, C]
        
        Xev = Xe[..., edges, :]  # [nnz, C]
        Xv = torch_scatter.scatter(Xev, vertex, dim=-2, reduce=self.aggr, dim_size=N)  # [N, C]

        X = Xv

        Xi = (1 - self.alpha) * X + self.alpha * X0
        X = (1 - beta) * Xi + beta * self.W(Xi)

        return X, Xe


class EquivSetGNN(nn.Module):
    def __init__(self, num_features, num_targets, args):

        super().__init__()
        act = {'Id': nn.Identity(), 'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.act = act[args.activation]
        self.dropout = args.dropout
        self.norm = args.normalization

        self.in_channels = num_features
        self.hidden_channels = args.MLP_hidden
        self.output_channels = num_targets

        self.mlp1_layers = args.MLP_num_layers
        self.mlp2_layers = args.MLP_num_layers if args.MLP2_num_layers < 0 else args.MLP2_num_layers
        self.mlp3_layers = args.MLP_num_layers if args.MLP3_num_layers < 0 else args.MLP3_num_layers
        self.num_layers = args.All_num_layers
        self.edconv_type = args.edconv_type

        self.lin_in = torch.nn.Linear(num_features, args.MLP_hidden)
        if args.edconv_type == 'EquivSet':
            self.conv = EquivSetConv(args.MLP_hidden, args.MLP_hidden, mlp1_layers=self.mlp1_layers,
                                     mlp2_layers=self.mlp2_layers, mlp3_layers=self.mlp3_layers,
                                     alpha=args.alpha, aggr=args.aggregate, dropout=self.dropout,
                                     normalization=self.norm, input_norm=args.AllSet_input_norm)
        elif args.edconv_type == 'JumpLink':
            self.conv = JumpLinkConv(args.MLP_hidden, args.MLP_hidden, mlp_layers=self.mlp1_layers,
                                    alpha=args.alpha, aggr=args.aggregate)
        else:
            raise ValueError(f'Unsupported EDConv type: {args.edconv_type}')

        self.decoder = MLP(in_channels=args.MLP_hidden,
            hidden_channels=args.decoder_hidden,
            out_channels=self.output_channels,
            num_layers=args.decoder_num_layer,
            dropout=self.dropout,
            Normalization=self.norm,
            InputNorm=False)

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        self.conv.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, data):
        x = data.x
        V, E = data.hyperedge_index[0], data.hyperedge_index[1]
        x = F.relu(self.lin_in(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x0 = x
        for i in range(self.num_layers):
            x, e = self.conv(x, V, E, x0)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.decoder(x)
        return x, e
