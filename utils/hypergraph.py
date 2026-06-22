from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch

from utils.dataset_registry import build_domain_aliases as _build_domain_aliases

@dataclass
class SimpleHypergraph:
    num_nodes: int
    hyperedges: List[List[int]]
    x: torch.Tensor
    name: str
    domain: str
    dataset_name: str
    node_labels: torch.Tensor
    edge_labels: Optional[torch.Tensor]
    graph_label: Optional[torch.Tensor]
    node_train_mask: Optional[torch.Tensor]
    node_val_mask: Optional[torch.Tensor]
    node_test_mask: Optional[torch.Tensor]
    metadata: Dict

    def incidence_matrix(self) -> torch.Tensor:
        num_edges = len(self.hyperedges)
        incidence = torch.zeros(self.num_nodes, num_edges, dtype=self.x.dtype)
        for edge_index, nodes in enumerate(self.hyperedges):
            if not nodes:
                continue
            incidence[nodes, edge_index] = 1.0
        return incidence

    def edge_sizes(self) -> torch.Tensor:
        return torch.tensor([max(len(edge), 1) for edge in self.hyperedges], dtype=self.x.dtype)


def build_domain_aliases() -> Dict[str, str]:
    return _build_domain_aliases()


def iter_graphs(domains: Dict[str, List[SimpleHypergraph]], allowed_domains: Iterable[str] | None = None) -> List[SimpleHypergraph]:
    selected = set(allowed_domains) if allowed_domains is not None else None
    graphs: List[SimpleHypergraph] = []
    for domain_name, domain_graphs in domains.items():
        if selected is not None and domain_name not in selected:
            continue
        graphs.extend(domain_graphs)
    return graphs
