"""Pretext-task heads for HyperFounder V2 (规格书 §5).

Three tasks:
  1. EdgeReconHead        — edge MLM (§5.1): reconstruct mean-node-feature from masked edge embedding
  2. MembershipHead       — node/edge bilinear scorer (§5.2): score (n, e) membership triples
  3. EdgeContrastProjector — L2-normalised projector for symmetric edge dual-view contrast (§5.3)
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn


class EdgeReconHead(nn.Module):
    """2-layer GELU MLP decoder for §5.1 edge-MLM.

    Target: the original mean-node-feature vector of the hyperedge (shape [in_dim]).
    """

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, out_dim),
        )

    def forward(self, edge_emb: torch.Tensor) -> torch.Tensor:
        return self.decoder(edge_emb)


class MembershipHead(nn.Module):
    """Bilinear scorer for §5.2 (n, e) membership triples.

    score = N_v^T W E_e → scalar per pair.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(self, node_emb: torch.Tensor, edge_emb: torch.Tensor) -> torch.Tensor:
        return self.W(node_emb, edge_emb).squeeze(-1)


class EdgeContrastProjector(nn.Module):
    """2-layer projector for §5.3 edge dual-view symmetric InfoNCE.

    Outputs L2-normalised embeddings (shape [E, proj_dim]).  Default proj_dim=128
    (consistent with most contrastive baselines).
    """

    def __init__(self, hidden_dim: int, proj_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)
