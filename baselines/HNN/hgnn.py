import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from typing import Optional
from baselines.HNN.utils import zeros,glorot

class HypergraphConv(MessagePassing):

    def __init__(self, in_channels, out_channels, symdegnorm=False, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HypergraphConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.symdegnorm = symdegnorm

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = nn.Parameter(
                torch.Tensor(in_channels, heads * out_channels))
            self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix :math:`\mathbf{X}`
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Sparse hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges,device=x.device)

        x = torch.matmul(x, self.weight)

        alpha = None
        if self.use_attention:
            assert num_edges <= num_edges
            x = x.view(-1, self.heads, self.out_channels)
            x_i, x_j = x[hyperedge_index[0]], x[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if not self.symdegnorm:
            D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                            hyperedge_index[0], dim=0, dim_size=num_nodes)
            D = 1.0 / D
            D[D == float("inf")] = 0

            B = scatter_add(x.new_ones(hyperedge_index.size(1),device=x.device),
                            hyperedge_index[1], dim=0, dim_size=num_edges)  
            B = 1.0 / B
            B[B == float("inf")] = 0
            
            self.flow = 'source_to_target'
            edge_embed = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                                 size=(num_nodes, num_edges))
            self.flow = 'target_to_source'
            out = self.propagate(hyperedge_index, x=edge_embed, norm=D, alpha=alpha,size=(num_nodes, num_edges))
            
        else:  # this correspond to HGNN
            D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                            hyperedge_index[0], dim=0, dim_size=num_nodes)
            D = 1.0 / D**(0.5)
            D[D == float("inf")] = 0

            B = scatter_add(x.new_ones(hyperedge_index.size(1),device=x.device),
                            hyperedge_index[1], dim=0, dim_size=num_edges)
            B = 1.0 / B
            B[B == float("inf")] = 0

            x = D.unsqueeze(-1)*x
            self.flow = 'source_to_target'
            edge_embed = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                                 size=(num_nodes, num_edges))

            self.flow = 'target_to_source'
            out = self.propagate(hyperedge_index,x=edge_embed, norm=D, alpha=alpha,size=(num_nodes, num_edges))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
            edge_embed = edge_embed.view(-1,self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            edge_embed = edge_embed.mean(dim=-1)

        if self.bias is not None:
            out = out + self.bias

        return out,edge_embed 

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels
        
        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class HCHA(nn.Module):

    def __init__(self, num_features, num_targets, args):
        super(HCHA, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout  # Note that default is 0.6
        self.symdegnorm = args.HCHA_symdegnorm
        self.hidden_dim = args.MLP_hidden

#       Note that add dropout to attention is default in the original paper
        self.convs = nn.ModuleList()
        if self.num_layers == 1:
            self.convs.append(HypergraphConv(num_features,
                               num_targets, self.symdegnorm))
        else:
            self.convs.append(HypergraphConv(num_features,
                                            self.hidden_dim, self.symdegnorm))
            for _ in range(self.num_layers-2):
                self.convs.append(HypergraphConv(
                    self.hidden_dim, self.hidden_dim, self.symdegnorm))
            # Output heads is set to 1 as default
            self.convs.append(HypergraphConv(
                self.hidden_dim, num_targets, self.symdegnorm))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):

        # regular node classification

        x = data.x
        edge_index = data.hyperedge_index

        for i, conv in enumerate(self.convs[:-1]):
            x , e = conv(x,edge_index) 
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x,e = self.convs[-1](x, edge_index)

        return x,e

    @torch.no_grad()
    def predict(self,data):
        self.eval()
        return self.forward(data)