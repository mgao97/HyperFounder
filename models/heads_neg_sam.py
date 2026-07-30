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


class TaskHeadsNegSam(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        input_dim: int,
        num_domains: int,
        projection_dim: int = 64,
    ):
        super().__init__()
        self.masked_node_decoder = nn.Linear(hidden_dim, input_dim)
        self.edge_size_regressor = nn.Linear(hidden_dim, 1)
        self.node_projector = ProjectionHead(hidden_dim, projection_dim)
        self.domain_classifier = nn.Linear(hidden_dim, num_domains)
        self.hyperedge_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.membership_scorer = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.subgraph_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
