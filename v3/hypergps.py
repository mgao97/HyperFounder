"""HyperGPS layer: the GFSE GPS backbone adapted to hypergraphs.

Each layer runs a local hypergraph MPNN and a global biased self-attention in
parallel, then aggregates (Eq. 2 of GFSE). The ONLY change vs GFSE is that
(i) the MPNN is a clique-free, size-normalized hypergraph convolution and
(ii) the relative random-walk encoding R comes from encoding.HypergraphRandomWalkPE.
The biased-attention mechanism a'_{ij} = softmax(a_{ij} + Linear(R_{ij})) is kept
byte-for-byte identical to GFSE, because R has the same (N,N,d) shape as in the
graph case.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperMPNN(nn.Module):
    """Clique-free, hyperedge-size-normalized hypergraph message passing."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.pre = nn.Linear(dim, dim)
        self.post = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        # x: (N,d) float32 ; H: (N,E) float32 (0/1)
        h = self.act(self.pre(x))                 # (N,d)
        he_msg = H.T @ h                          # (E,d) node->hyperedge
        sizes = H.sum(0, keepdim=True).clamp(min=1.0).T  # (E,1)
        he_msg = he_msg / sizes                  # divide by hyperedge size
        node_msg = H @ he_msg                    # (N,d) hyperedge->node
        return self.post(self.drop(node_msg))


class BiasedAttention(nn.Module):
    """Multi-head self-attention with relative random-walk bias (GFSE Eq.2)."""

    def __init__(self, dim: int, num_heads: int, rw_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.R_bias = nn.Linear(rw_dim, 1)  # R_ij (d) -> scalar, shared across heads
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(N, self.num_heads, self.head_dim).transpose(0, 1) for t in qkv]
        attn = (q @ k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (H,N,N)
        bias = self.R_bias(R).squeeze(-1)  # (N,N)
        attn = attn + bias.unsqueeze(0)     # broadcast over heads
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(0, 1).contiguous().view(N, -1)  # (N,d)
        return self.proj(out)


class HyperGPSLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, rw_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mpnn = HyperMPNN(dim, dropout)
        self.attn = BiasedAttention(dim, num_heads, rw_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2 * dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, R: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        x_m = self.norm1(x + self.mpnn(x, H))
        x_t = self.norm2(x_m + self.attn(x_m, R))
        return x_t + self.mlp(x_t)
