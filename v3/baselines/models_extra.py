"""
Additional hypergraph baseline models for the v3 transductive trainer.

- HGNNP  (HGNN+): hypergraph attention. Reuses the (now fixed) HypergraphConv
  from baselines/HNN/hgnn.py with use_attention=True. The original attention
  path in that module indexed node features with edge indices (invalid for the
  [node, edge] incidence format) -- it has been corrected to derive attention
  coefficients from node features, grouped per hyperedge.

- UniGCN: a clean GCN-style hypergraph convolution (mean first-aggregate +
  symmetric node/edge-degree normalization, no (1+eps) skip connection),
  following the UniGCN instantiation of the UniGNN framework (Huang et al., 2021).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from baselines.HNN.hgnn import HypergraphConv


# ---------------------------------------------------------------------------
# HGNN+  (HGNN with hypergraph attention)
# ---------------------------------------------------------------------------
class HGNNP(nn.Module):
    """HGNN+ : HGNN (HCHA) with attention enabled on every HypergraphConv layer."""

    def __init__(self, num_features, num_targets, args):
        super().__init__()
        self.num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.hidden_dim = args.MLP_hidden
        self.heads = getattr(args, "heads", 8)
        self.convs = nn.ModuleList()

        if self.num_layers == 1:
            self.convs.append(
                HypergraphConv(
                    num_features, num_targets,
                    symdegnorm=True, use_attention=True,
                    heads=self.heads, concat=False,
                )
            )
        else:
            self.convs.append(
                HypergraphConv(
                    num_features, self.hidden_dim,
                    symdegnorm=True, use_attention=True,
                    heads=self.heads, concat=False,
                )
            )
            for _ in range(self.num_layers - 2):
                self.convs.append(
                    HypergraphConv(
                        self.hidden_dim, self.hidden_dim,
                        symdegnorm=True, use_attention=True,
                        heads=self.heads, concat=False,
                    )
                )
            self.convs.append(
                HypergraphConv(
                    self.hidden_dim, num_targets,
                    symdegnorm=True, use_attention=True,
                    heads=self.heads, concat=False,
                )
            )

    def reset_parameters(self):
        for c in self.convs:
            c.reset_parameters()

    def forward(self, data):
        x = data.x
        ei = data.hyperedge_index
        for conv in self.convs[:-1]:
            x, _ = conv(x, ei)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x, _ = self.convs[-1](x, ei)
        return x, None


# ---------------------------------------------------------------------------
# UniGCN  (GCN-style hypergraph convolution)
# ---------------------------------------------------------------------------
class UniGCNConv(nn.Module):
    """Single UniGCN layer.

    Aggregation (node -> edge): mean pooling over the nodes of each hyperedge.
    Propagation (edge -> node): symmetric normalization by node degree and
    hyperedge size, with no residual (1+eps) skip connection -- i.e. the
    hypergraph analogue of GCN applied to the clique expansion.
    """

    def __init__(self, args, in_channels, out_channels):
        super().__init__()
        self.W = nn.Linear(in_channels, out_channels, bias=True)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        X = self.W(X)

        # first aggregate: mean over nodes per hyperedge
        E = int(edges.max().item()) + 1
        Xe = scatter(X[vertex], edges, dim=0, reduce="mean", dim_size=E)  # [E, C]

        # normalize by hyperedge size (sqrt)
        esz = scatter(
            torch.ones_like(edges, dtype=X.dtype), edges,
            dim=0, reduce="sum", dim_size=E,
        )
        inv_esz = (esz + 1e-12).clamp(min=1e-12).pow(-0.5)
        Xev = Xe[edges] * inv_esz[edges].unsqueeze(-1)

        # second aggregate: sum over edges per node, normalize by node degree (sqrt)
        Xv = scatter(Xev, vertex, dim=0, reduce="sum", dim_size=N)
        d = scatter(
            torch.ones_like(vertex, dtype=X.dtype), vertex,
            dim=0, reduce="sum", dim_size=N,
        )
        inv_d = (d + 1e-12).clamp(min=1e-12).pow(-0.5)
        Xv = Xv * inv_d.unsqueeze(-1)
        return Xv, Xe


class UniGCN(nn.Module):
    """UniGCN stack built from UniGCNConv layers."""

    def __init__(self, num_features, num_targets, args):
        super().__init__()
        self.num_layers = args.All_num_layers
        self.hidden_dim = args.MLP_hidden
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)
        self.act = nn.ReLU()

        if self.num_layers == 1:
            self.convs = nn.ModuleList()
            self.conv_out = UniGCNConv(args, num_features, num_targets)
        else:
            self.conv_out = UniGCNConv(args, self.hidden_dim, num_targets)
            self.convs = nn.ModuleList(
                [UniGCNConv(args, num_features, self.hidden_dim)]
                + [UniGCNConv(args, self.hidden_dim, self.hidden_dim)
                   for _ in range(self.num_layers - 2)]
            )

    def reset_parameters(self):
        for c in self.convs:
            c.reset_parameters()
        self.conv_out.reset_parameters()

    def forward(self, data):
        X = data.x
        V, E = data.hyperedge_index[0], data.hyperedge_index[1]
        X = self.input_drop(X)
        for conv in self.convs:
            X, _ = conv(X, V, E)
            X = self.act(X)
            X = self.dropout(X)
        X, Xe = self.conv_out(X, V, E)
        return X, Xe
