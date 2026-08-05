"""
Subgraph Quality Scoring Module for Challenge 2.

Computes quality scores for sampled sub-hypergraphs to determine:
- Task routing decisions
- Hard negative bank eligibility
- Supervision intensity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class SubgraphQualityMeta:
    """Quality metadata for a sub-hypergraph."""
    num_nodes: int
    num_edges: int
    incidence_nnz: int
    component_ratio: float
    overlap_density: float
    cardinality_mean: float
    cardinality_std: float
    validity_flag: bool
    quality_score: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "incidence_nnz": self.incidence_nnz,
            "component_ratio": self.component_ratio,
            "overlap_density": self.overlap_density,
            "cardinality_mean": self.cardinality_mean,
            "cardinality_std": self.cardinality_std,
            "validity_flag": float(self.validity_flag),
            "quality_score": self.quality_score,
        }


class SubgraphQualityScorer(nn.Module):
    """
    Computes quality score for sub-hypergraphs.
    
    Quality is determined by:
    - Structural validity (connected, sufficient edges)
    - Size appropriateness
    - Overlap density
    - Component structure
    """

    def __init__(
        self,
        min_nodes: int = 4,
        min_edges: int = 2,
        max_component_ratio: float = 0.9,
        min_overlap_density: float = 0.05,
    ):
        super().__init__()
        self.min_nodes = min_nodes
        self.min_edges = min_edges
        self.max_component_ratio = max_component_ratio
        self.min_overlap_density = min_overlap_density

    def compute_quality(
        self,
        num_nodes: int,
        num_edges: int,
        incidence: torch.Tensor,
        hyperedge_sizes: Optional[torch.Tensor] = None,
    ) -> SubgraphQualityMeta:
        """
        Compute quality score and metadata for a sub-hypergraph.
        
        Args:
            num_nodes: Number of nodes
            num_edges: Number of edges
            incidence: Incidence matrix (num_nodes, num_edges)
            hyperedge_sizes: Optional tensor of edge cardinalities
        
        Returns:
            SubgraphQualityMeta with quality score and statistics
        """
        if incidence.is_sparse:
            dense_inc = incidence.to_dense()
        else:
            dense_inc = incidence
        
        # Basic counts
        incidence_nnz = int((dense_inc > 0).sum().item())
        
        # Validity check
        validity_flag = (
            num_nodes >= self.min_nodes
            and num_edges >= self.min_edges
            and incidence_nnz > 0
        )
        
        # Component ratio: connected components relative to size
        if num_nodes > 0 and num_edges > 0:
            # Simple connected component estimation using union-find-like approach
            # Count nodes that have any incident edge
            nodes_with_edges = (dense_inc.sum(dim=1) > 0).sum().item()
            component_ratio = nodes_with_edges / num_nodes if num_nodes > 0 else 0.0
        else:
            component_ratio = 0.0
        
        # Overlap density: average pairwise edge overlap
        if num_edges >= 2:
            edge_overlaps = []
            for i in range(num_edges):
                for j in range(i + 1, num_edges):
                    overlap = (dense_inc[:, i] * dense_inc[:, j]).sum().item()
                    edge_overlaps.append(overlap)
            overlap_density = sum(edge_overlaps) / len(edge_overlaps) if edge_overlaps else 0.0
            overlap_density = min(overlap_density / max(num_nodes, 1), 1.0)
        else:
            overlap_density = 0.0
        
        # Cardinality statistics
        if hyperedge_sizes is not None and hyperedge_sizes.numel() > 0:
            cardinality_mean = hyperedge_sizes.float().mean().item()
            cardinality_std = hyperedge_sizes.float().std().item()
        else:
            cardinalities = dense_inc.sum(dim=0)
            cardinality_mean = cardinalities.float().mean().item() if cardinalities.numel() > 0 else 0.0
            cardinality_std = cardinalities.float().std().item() if cardinalities.numel() > 1 else 0.0
        
        # Compute quality score
        quality_score = self._compute_quality_score(
            validity_flag=validity_flag,
            num_nodes=num_nodes,
            num_edges=num_edges,
            component_ratio=component_ratio,
            overlap_density=overlap_density,
            incidence_nnz=incidence_nnz,
        )
        
        return SubgraphQualityMeta(
            num_nodes=num_nodes,
            num_edges=num_edges,
            incidence_nnz=incidence_nnz,
            component_ratio=component_ratio,
            overlap_density=overlap_density,
            cardinality_mean=cardinality_mean,
            cardinality_std=cardinality_std,
            validity_flag=validity_flag,
            quality_score=quality_score,
        )

    def _compute_quality_score(
        self,
        validity_flag: bool,
        num_nodes: int,
        num_edges: int,
        component_ratio: float,
        overlap_density: float,
        incidence_nnz: int,
    ) -> float:
        """
        Combine factors into a single quality score.
        
        Returns:
            quality_score in [0, 1]
        """
        if not validity_flag:
            return 0.0
        
        # Size score: prefer medium-sized subgraphs
        # Too small: low score, Too large: moderate score
        size_score = min(num_nodes / 32.0, 1.0) * min(num_edges / 8.0, 1.0)
        size_score = min(size_score, 1.0)
        
        # Connectedness score: prefer connected subgraphs
        connectivity_score = min(component_ratio / self.max_component_ratio, 1.0)
        
        # Overlap score: prefer some overlap
        overlap_score = min(overlap_density / max(self.min_overlap_density, 0.1), 1.0)
        
        # Density score: incidence density
        max_nnz = num_nodes * num_edges if num_nodes > 0 and num_edges > 0 else 1
        density_score = incidence_nnz / max(max_nnz, 1)
        
        # Weighted combination
        quality = (
            0.25 * size_score +
            0.30 * connectivity_score +
            0.25 * overlap_score +
            0.20 * density_score
        )
        
        return min(max(quality, 0.0), 1.0)

    def forward(
        self,
        num_nodes: int,
        num_edges: int,
        incidence: torch.Tensor,
        hyperedge_sizes: Optional[torch.Tensor] = None,
    ) -> SubgraphQualityMeta:
        """Compute quality metadata."""
        return self.compute_quality(num_nodes, num_edges, incidence, hyperedge_sizes)


def get_task_routing_decision(
    quality_meta: SubgraphQualityMeta,
    tau_hyperedge: float = 0.55,
    tau_motif: float = 0.70,
    tau_membership: float = 0.40,
    tau_hard_negative: float = 0.40,
    min_membership_nodes: int = 2,
) -> Dict[str, bool]:
    """
    Determine which tasks should be applied based on quality score.
    
    Routing rules:
    - quality >= tau_motif -> motif, community
    - quality >= tau_hyperedge -> hyperedge_recon, contrastive
    - quality >= tau_membership -> membership
    - quality < tau_hard_negative -> hard_negative (weak but valid)
    - not valid -> exclude
    
    Returns:
        Dictionary with routing decisions for each task
    """
    quality = quality_meta.quality_score
    valid = quality_meta.validity_flag
    num_nodes = quality_meta.num_nodes
    
    routing = {
        "valid": valid,
        "membership": False,
        "hyperedge_recon": False,
        "contrastive": False,
        "motif": False,
        "community": False,
        "structure_discrimination": False,
        "hard_negative": False,
        "exclude": not valid,
    }
    
    if not valid:
        return routing
    
    # Hard negative: weak but valid
    if quality < tau_hard_negative:
        routing["hard_negative"] = True
        routing["exclude"] = False
        return routing
    
    # Membership task: needs minimum nodes
    if quality >= tau_membership and num_nodes >= min_membership_nodes:
        routing["membership"] = True
    
    # Hyperedge reconstruction: moderate quality
    if quality >= tau_hyperedge:
        routing["hyperedge_recon"] = True
        routing["contrastive"] = True
    
    # Motif/community: high quality
    if quality >= tau_motif:
        routing["motif"] = True
        routing["community"] = True
        routing["structure_discrimination"] = True
    
    return routing


def aggregate_batch_quality(
    quality_metas: list[SubgraphQualityMeta],
) -> Dict[str, float]:
    """
    Aggregate quality statistics for a batch of sub-hypergraphs.
    
    Returns:
        Dictionary with aggregated statistics
    """
    if not quality_metas:
        return {
            "num_valid": 0,
            "num_invalid": 0,
            "num_hard_negatives": 0,
            "num_excluded": 0,
            "avg_quality_score": 0.0,
            "avg_nodes": 0.0,
            "avg_edges": 0.0,
        }
    
    total = len(quality_metas)
    num_valid = sum(1 for m in quality_metas if m.validity_flag)
    num_invalid = total - num_valid
    num_hard_neg = sum(1 for m in quality_metas if 0 < m.quality_score < 0.4)
    num_excluded = sum(1 for m in quality_metas if not m.validity_flag)
    
    avg_quality = sum(m.quality_score for m in quality_metas) / total
    avg_nodes = sum(m.num_nodes for m in quality_metas) / total
    avg_edges = sum(m.num_edges for m in quality_metas) / total
    
    return {
        "num_valid": num_valid,
        "num_invalid": num_invalid,
        "num_hard_negatives": num_hard_neg,
        "num_excluded": num_excluded,
        "avg_quality_score": avg_quality,
        "avg_nodes": avg_nodes,
        "avg_edges": avg_edges,
        "valid_ratio": num_valid / total,
    }
