import torch
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

class HNHNConv(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, nonlinear_inbetween=True,
                 concat=True, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HNHNConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.nonlinear_inbetween = nonlinear_inbetween

        self.heads = heads
        self.concat = True

        self.weight_v2e = Linear(in_channels, hidden_channels)
        self.weight_e2v = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_v2e.reset_parameters()
        self.weight_e2v.reset_parameters()

    def forward(self, x, data):

        hyperedge_index = data.hyperedge_index
        hyperedge_weight = None
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = self.weight_v2e(x)

        x = data.D_v_beta.unsqueeze(-1) * x

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=data.D_e_beta_inv,
                             size=(num_nodes, num_edges))
        
        edge_embed=out 

        if self.nonlinear_inbetween:
            out = F.relu(out)
        
        # sanity check
        out = torch.squeeze(out, dim=1)
        
        out = self.weight_e2v(out)
        
        out = data.D_e_alpha.unsqueeze(-1) * out

        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=data.D_v_alpha_inv,
                             size=(num_nodes,num_edges))
        
        return out, edge_embed

    def message(self, x_j, norm_i):

        out = norm_i.view(-1, 1) * x_j

        return out

    def __repr__(self):
        return "{}({}, {}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.hidden_channels, self.out_channels)

class HNHN(nn.Module):

    def __init__(self, num_features, num_targets, args):
        super(HNHN, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.hidden_dim = args.MLP_hidden
        
        self.convs = nn.ModuleList()
        # two cases
        if self.num_layers == 1:
            self.convs.append(HNHNConv(num_features, self.hidden_dim, num_targets,
                                       nonlinear_inbetween=args.HNHN_nonlinear_inbetween))
        else:
            self.convs.append(HNHNConv(num_features, self.hidden_dim, self.hidden_dim,
                                       nonlinear_inbetween=args.HNHN_nonlinear_inbetween))
            for _ in range(self.num_layers - 2):
                self.convs.append(HNHNConv(self.hidden_dim, self.hidden_dim, self.hidden_dim,
                                           nonlinear_inbetween=args.HNHN_nonlinear_inbetween))
            self.convs.append(HNHNConv(self.hidden_dim, self.hidden_dim, num_targets,
                                       nonlinear_inbetween=args.HNHN_nonlinear_inbetween))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):

        x = data.x
        
        if self.num_layers == 1:
            conv = self.convs[0]
            x,e = conv(x, data)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                x,e = conv(x, data)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x,e = self.convs[-1](x, data)

        return x,e