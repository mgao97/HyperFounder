from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HypergraphData:
    node_features: torch.Tensor
    edge_features: torch.Tensor
    incidence_matrix: torch.Tensor
    node_labels: torch.Tensor | None
    domain_id: int
    feature_type: str

