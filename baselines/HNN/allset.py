import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.typing import Adj, Size, OptTensor
from baselines.HNN.mlp import MLP
from baselines.HNN.utils import zeros,glorot

class PMA(MessagePassing):

    _alpha: OptTensor

    def __init__(self, in_channels, hid_dim,
                 out_channels, num_layers, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.0, bias=False, **kwargs):
        #         kwargs.setdefault('aggr', 'add')
        super(PMA, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.hidden = hid_dim // heads
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = 0.
        self.aggr = 'add'

        self.lin_K = nn.Linear(in_channels, self.heads*self.hidden)
        self.lin_V = nn.Linear(in_channels, self.heads*self.hidden)
        self.att_r = nn.Parameter(torch.Tensor(
            1, heads, self.hidden))  # Seed vector
        self.rFF = MLP(in_channels=self.heads*self.hidden,
                       hidden_channels=self.heads*self.hidden,
                       out_channels=out_channels,
                       num_layers=num_layers,
                       dropout=.0, Normalization='None',)
        self.ln0 = nn.LayerNorm(self.heads*self.hidden)
        self.ln1 = nn.LayerNorm(self.heads*self.hidden)

        self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        #         glorot(self.lin_l.weight)
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)
        self.rFF.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
#         glorot(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
#         zeros(self.bias)

    def forward(self, x, edge_index: Adj,
                size: Size = None, return_attention_weights=None):
        
        H, C = self.heads, self.hidden

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_K = self.lin_K(x).view(-1, H, C)
            x_V = self.lin_V(x).view(-1, H, C)
            alpha_r = (x_K * self.att_r).sum(dim=-1)

        out = self.propagate(edge_index, x=x_V,
                             alpha=alpha_r, aggr=self.aggr)

        alpha = self._alpha
        self._alpha = None

#         Note that in the original code of GMT paper, they do not use additional W^O to combine heads.
#         This is because O = softmax(QK^T)V and V = V_in*W^V. So W^O can be effectively taken care by W^V!!!
        out += self.att_r  # This is Seed + Multihead
        # concat heads then LayerNorm. Z (rhs of Eq(7)) in GMT paper.
        out = self.ln0(out.view(-1, self.heads * self.hidden))
        # rFF and skip connection. Lhs of eq(7) in GMT paper.
        out = self.ln1(out+F.relu(self.rFF(out)))

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j, alpha_j,
                index, ptr,
                size_j):
        #         ipdb.set_trace()
        alpha = alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, index.max()+1)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def aggregate(self, inputs, index,
                  dim_size=None, aggr='add'):

        if aggr is None:
            raise ValueError("aggr was not passed!")
        return scatter(inputs, index, dim=self.node_dim, reduce=aggr)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class HalfNLHconv(MessagePassing):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 dropout,
                 Normalization='bn',
                 InputNorm=False,
                 heads=1,
                 attention=True
                 ):
        super(HalfNLHconv, self).__init__()

        self.attention = attention
        self.dropout = dropout

        if self.attention:
            self.prop = PMA(in_dim, hid_dim, out_dim, num_layers, heads=heads)
        else:
            if num_layers > 0:
                self.f_enc = MLP(in_dim, hid_dim, hid_dim, num_layers, dropout, Normalization, InputNorm)
                self.f_dec = MLP(hid_dim, hid_dim, out_dim, num_layers, dropout, Normalization, InputNorm)
            else:
                self.f_enc = nn.Identity()
                self.f_dec = nn.Identity()

    def reset_parameters(self):

        if self.attention:
            self.prop.reset_parameters()
        else:
            if not (self.f_enc.__class__.__name__ != 'Identity'):
                self.f_enc.reset_parameters()
            if not (self.f_dec.__class__.__name__ != 'Identity'):
                self.f_dec.reset_parameters()

    def forward(self, x, edge_index, norm, aggr='add'):
        
        if self.attention:
            x = self.prop(x, edge_index)
        else:
            x = F.relu(self.f_enc(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.propagate(edge_index, x=x, norm=norm, aggr=aggr)
            x = F.relu(self.f_dec(x))
            
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def aggregate(self, inputs, index,
                  dim_size=None, aggr=None):
        aggr = aggr if aggr is not None else "add"
        if aggr is None:
            raise ValueError("aggr was not passed!")
        return scatter(inputs, index, dim=self.node_dim, reduce=aggr)


class SetGNN(nn.Module):
    def __init__(self, num_features, num_targets, args, norm=None):
        super(SetGNN, self).__init__()

        self.All_num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.aggr = args.aggregate
        self.NormLayer = args.normalization
        self.InputNorm = args.deepset_input_norm
        self.GPR = args.GPR
        self.LearnMask = args.LearnMask
        self.decoder_hidden = args.decoder_hidden
        self.decoder_num_layers = args.decoder_num_layers

        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()
        self.bnV2Es = nn.ModuleList()
        self.bnE2Vs = nn.ModuleList()

        if self.LearnMask:
            self.Importance = nn.Parameter(torch.ones(norm.size()))

        if self.All_num_layers == 0:
            self.decoder = MLP(in_channels=num_features,
                                  hidden_channels=self.decoder_hidden,
                                  out_channels=num_targets,
                                  num_layers=self.decoder_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False)
        else:
            self.V2EConvs.append(HalfNLHconv(in_dim=num_features,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA))
            self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
            self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA))
            self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
            for _ in range(self.All_num_layers-1):
                self.V2EConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA))
                self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
                self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA))
                self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
            if self.GPR:
                self.MLP = MLP(in_channels=num_features,
                               hidden_channels=args.MLP_hidden,
                               out_channels=args.MLP_hidden,
                               num_layers=args.MLP_num_layers,
                               dropout=self.dropout,
                               Normalization=self.NormLayer,
                               InputNorm=False)
                self.GPRweights = nn.Linear(self.All_num_layers+1, 1, bias=False)
                self.decoder = MLP(in_channels=args.MLP_hidden,
                                      hidden_channels=self.decoder_hidden,
                                      out_channels=num_targets,
                                      num_layers=self.decoder_num_layers,
                                      dropout=self.dropout,
                                      Normalization=self.NormLayer,
                                      InputNorm=False)
            else:
                self.decoder = MLP(in_channels=args.MLP_hidden,
                                      hidden_channels=self.decoder_hidden,
                                      out_channels=num_targets,
                                      num_layers=self.decoder_num_layers,
                                      dropout=self.dropout,
                                      Normalization=self.NormLayer,
                                      InputNorm=False)

    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        self.decoder.reset_parameters()
        if self.GPR:
            self.MLP.reset_parameters()
            self.GPRweights.reset_parameters()
        if self.LearnMask:
            nn.init.ones_(self.Importance)

    def forward(self, data):

        x, edge_index, norm = data.x, data.hyperedge_index, data.norm
        if self.LearnMask:
            norm = self.Importance*norm
        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack(
            [edge_index[1], edge_index[0]], dim=0)
        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                e = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr)) 
                x = F.dropout(e, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                x = F.relu(x)
                xs.append(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
            x = self.decoder(x)
        else:
            x = F.dropout(x, p=0.2, training=self.training) # Input dropout
            for i, _ in enumerate(self.V2EConvs):
                e = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
                x = F.dropout(e, p=self.dropout, training=self.training)
                x = F.relu(self.E2VConvs[i](
                    x, reversed_edge_index, norm, self.aggr))
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.decoder(x)

        return x,e

