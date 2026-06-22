from __future__ import annotations

import torch
from torch import nn


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.net(x)
        return nn.functional.normalize(projected, dim=-1)


class TaskHeads(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        node_classes: int,
        motif_classes: int,
        community_classes: int,
        prototype_classes: int,
        projection_dim: int = 64,
    ):
        super().__init__()
        self.node_head = MLPHead(hidden_dim, node_classes)
        self.edge_head = MLPHead(hidden_dim, 1)
        self.motif_head = MLPHead(hidden_dim, motif_classes)
        self.community_head = MLPHead(hidden_dim, community_classes)
        self.graph_projector = ProjectionHead(hidden_dim, projection_dim)
        self.prototype_head = MLPHead(hidden_dim, prototype_classes)
        self.structure_projection = nn.Linear(hidden_dim, hidden_dim)
