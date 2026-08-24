"""
Domain Confidence Scoring Module for Cross-Domain Hypergraph Learning.

This module implements confidence-based selective alignment:
1. Node confidence scoring based on structural validity
2. Edge confidence scoring based on structural properties
3. Selective routing rules for alignment

Reference: Challenge 1 Solution Spec, Part C
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeConfidenceScorer(nn.Module):
    """
    Computes confidence scores for nodes to determine if they should participate in alignment.
    
    High confidence nodes:
    - Have sufficient incident hyperedges
    - Have good local overlap density
    - Show consistency across augmentations
    
    Low confidence nodes:
    - Domain-unique structures
    - Isolated or peripheral nodes
    - Small subgraphs with few connections
    """

    def __init__(
        self,
        hidden_dim: int,
        tau_align: float = 0.6,
        tau_low: float = 0.4,
    ):
        super().__init__()
        self.tau_align = tau_align
        self.tau_low = tau_low

        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def compute_structural_confidence(
        self,
        num_nodes: int,
        incidence: torch.Tensor,
        node_degrees: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute structural confidence based on hypergraph properties.
        
        Args:
            num_nodes: Number of nodes
            incidence: Incidence matrix (num_nodes, num_edges)
            node_degrees: Optional precomputed node degrees
        
        Returns:
            Confidence scores (num_nodes,)
        """
        if incidence.is_sparse:
            s = incidence.coalesce()
            idx = s.indices()
            row, col = idx[0], idx[1]  # row=node, col=edge
            device = incidence.device
        else:
            device = incidence.device
            nnz = incidence.nonzero(as_tuple=False)
            row, col = nnz[:, 0], nnz[:, 1]

        # Node degree score (normalized)
        if node_degrees is None:
            node_degrees = torch.zeros(num_nodes, device=device).index_add(
                0, row, torch.ones(row.numel(), device=device)
            )
        degree_score = (node_degrees / max(node_degrees.max().item(), 1.0)).clamp(0, 1).to(device)

        # Overlap score: for each node, count shared nodes across all pairs of
        # its incident hyperedges. Computed from the sparse COO structure only
        # (no dense materialisation), so it is safe on large hypergraphs.
        overlap_score = torch.zeros(num_nodes, device=device)
        if col.numel() > 0:
            # Build edge -> member-nodes mapping once (sort by edge id).
            e_order = torch.argsort(col, stable=True)
            col_e, row_e = col[e_order], row[e_order]
            num_edges_local = int(col.max().item()) + 1
            _, e_counts = torch.unique_consecutive(col_e, return_counts=True)
            e_offsets = torch.cat([
                torch.zeros(1, dtype=torch.long, device=device), e_counts.cumsum(0)
            ])
            # Cache member sets per edge to avoid re-scanning.
            edge_members_cache = {}
            def get_members(e):
                if e not in edge_members_cache:
                    st, en = int(e_offsets[e]), int(e_offsets[e + 1])
                    edge_members_cache[e] = set(row_e[st:en].tolist())
                return edge_members_cache[e]

            # Node -> incident edges mapping (sort by node id).
            order = torch.argsort(row, stable=True)
            row_s, col_s = row[order], col[order]
            _, counts = torch.unique_consecutive(row_s, return_counts=True)
            offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)])

            for i in range(num_nodes):
                if i >= offsets.size(0) - 1:
                    continue
                start, end = int(offsets[i]), int(offsets[i + 1])
                k = end - start
                if k < 2:
                    continue
                inc_edges = col_s[start:end].tolist()
                shared_count = 0
                for a in range(k):
                    ma = get_members(inc_edges[a])
                    if not ma:
                        continue
                    for b in range(a + 1, k):
                        shared_count += len(ma & get_members(inc_edges[b]))
                denom = max(k * (k - 1) // 2, 1)
                overlap_score[i] = shared_count / denom
            overlap_score = overlap_score.clamp(0, 1)

        # Validity score: node should have at least some connections
        validity_score = (node_degrees > 0).float().to(device)

        # Combine scores with weights
        confidence = (
            0.35 * degree_score +
            0.30 * overlap_score +
            0.35 * validity_score
        )

        return confidence.clamp(0, 1)

    def forward(
        self,
        node_emb: torch.Tensor,
        incidence: torch.Tensor,
        node_degrees: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute confidence scores for all nodes.
        
        Args:
            node_emb: Node embeddings (num_nodes, hidden_dim)
            incidence: Incidence matrix
            node_degrees: Optional precomputed node degrees
        
        Returns:
            Dictionary with confidence scores and routing masks
        """
        num_nodes = node_emb.size(0)

        # Embedding-based confidence
        emb_confidence = self.confidence_net(node_emb).squeeze(-1)

        # Structural confidence
        struct_confidence = self.compute_structural_confidence(num_nodes, incidence, node_degrees)

        # Combine embedding and structural confidence
        confidence = 0.5 * emb_confidence.detach() + 0.5 * struct_confidence

        # Routing masks
        align_mask = (confidence >= self.tau_align).float()
        neg_mask = ((confidence >= self.tau_low) & (confidence < self.tau_align)).float()
        exclude_mask = (confidence < self.tau_low).float()

        return {
            "confidence": confidence,
            "align_mask": align_mask,
            "neg_mask": neg_mask,
            "exclude_mask": exclude_mask,
            "num_confident": align_mask.sum().item(),
            "avg_confidence": confidence.mean().item(),
        }


class EdgeConfidenceScorer(nn.Module):
    """
    Computes confidence scores for hyperedges to determine alignment participation.
    
    High confidence edges:
    - Sufficient member count
    - Good member coherence
    - Structural validity
    
    Low confidence edges:
    - Degenerate (too small or too large)
    - Domain-specific artifacts
    """

    def __init__(
        self,
        hidden_dim: int,
        tau_align: float = 0.65,
        tau_low: float = 0.4,
    ):
        super().__init__()
        self.tau_align = tau_align
        self.tau_low = tau_low

        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def compute_structural_confidence(
        self,
        edge_emb: torch.Tensor,
        incidence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute edge structural confidence.
        
        Args:
            edge_emb: Edge embeddings (num_edges, hidden_dim)
            incidence: Incidence matrix (num_nodes, num_edges)
        
        Returns:
            Confidence scores (num_edges,)
        """
        if incidence.is_sparse:
            s = incidence.coalesce()
            idx = s.indices()
            row, col = idx[0], idx[1]  # row=node, col=edge
            device = incidence.device
        else:
            device = incidence.device
            nnz = incidence.nonzero(as_tuple=False)
            row, col = nnz[:, 0], nnz[:, 1]

        num_edges = edge_emb.size(0)
        confidence = torch.zeros(num_edges, device=edge_emb.device)

        # Edge -> member count (cardinality) from sparse COO.
        if col.numel() > 0:
            num_e_local = int(col.max().item()) + 1
            edge_card = torch.zeros(num_e_local, dtype=torch.long, device=device).index_add(
                0, col, torch.ones(col.numel(), dtype=torch.long, device=device)
            )
        else:
            edge_card = torch.zeros(num_edges, dtype=torch.long, device=device)

        # Overlap score: count, for each edge, how many *other* edges share at
        # least one node. Aggregated via node incidence (no dense materialisation,
        # no O(E^2) Python loop).
        overlap_share = torch.zeros(num_edges, device=device)
        if col.numel() > 0:
            e_order = torch.argsort(row, stable=True)
            row_e, col_e = row[e_order], col[e_order]
            _, n_counts = torch.unique_consecutive(row_e, return_counts=True)
            e_offsets = torch.cat([
                torch.zeros(1, dtype=torch.long, device=device), n_counts.cumsum(0)
            ])
            for n in range(row.max().item() + 1):
                if n >= e_offsets.size(0) - 1:
                    continue
                st, en = int(e_offsets[n]), int(e_offsets[n + 1])
                if en - st < 2:
                    continue
                inc = col_e[st:en]
                # every edge in `inc` shares node n with every other edge in `inc`
                for a in range(inc.numel()):
                    share_a = (inc != inc[a]).sum().item()  # number of other edges sharing n
                    overlap_share[inc[a]] += share_a
            # normalise by number of possible other edges
            denom = max(num_edges - 1, 1)
            overlap_share = (overlap_share.clamp(0, denom) / denom)

        cardinality = edge_card[:num_edges].tolist()
        for e_idx in range(num_edges):
            c = cardinality[e_idx]
            if 3 <= c <= 10:
                cardinality_score = 1.0
            elif (2 <= c < 3) or (10 < c <= 20):
                cardinality_score = 0.7
            else:
                cardinality_score = 0.3

            coherence = 0.5 if c >= 2 else 0.0
            overlap_score = float(overlap_share[e_idx].item())
            validity_score = 1.0 if c >= 2 else 0.0

            confidence[e_idx] = (
                0.20 * cardinality_score +
                0.25 * coherence +
                0.25 * overlap_score +
                0.30 * validity_score
            )

        return confidence.clamp(0, 1)

    def forward(
        self,
        edge_emb: torch.Tensor,
        incidence: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute confidence scores for all edges.
        
        Args:
            edge_emb: Edge embeddings (num_edges, hidden_dim)
            incidence: Incidence matrix
        
        Returns:
            Dictionary with confidence scores and routing masks
        """
        num_edges = edge_emb.size(0)

        # Embedding-based confidence
        emb_confidence = self.confidence_net(edge_emb).squeeze(-1)

        # Structural confidence
        struct_confidence = self.compute_structural_confidence(edge_emb, incidence)

        # Combine
        confidence = 0.5 * emb_confidence.detach() + 0.5 * struct_confidence

        # Routing masks
        align_mask = (confidence >= self.tau_align).float()
        neg_mask = ((confidence >= self.tau_low) & (confidence < self.tau_align)).float()
        exclude_mask = (confidence < self.tau_low).float()

        return {
            "confidence": confidence,
            "align_mask": align_mask,
            "neg_mask": neg_mask,
            "exclude_mask": exclude_mask,
            "num_confident": align_mask.sum().item(),
            "avg_confidence": confidence.mean().item(),
        }


class ConfidenceRouter(nn.Module):
    """
    Unified confidence router for both nodes and edges.
    Handles selective alignment based on confidence scores.
    """

    def __init__(
        self,
        hidden_dim: int,
        tau_node_align: float = 0.6,
        tau_node_low: float = 0.4,
        tau_edge_align: float = 0.65,
        tau_edge_low: float = 0.4,
    ):
        super().__init__()
        self.node_scorer = NodeConfidenceScorer(hidden_dim, tau_node_align, tau_node_low)
        self.edge_scorer = EdgeConfidenceScorer(hidden_dim, tau_edge_align, tau_edge_low)

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_emb: torch.Tensor,
        incidence: torch.Tensor,
        node_degrees: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute routing for nodes and edges.
        
        Args:
            node_emb: Node embeddings
            edge_emb: Edge embeddings
            incidence: Incidence matrix
            node_degrees: Optional node degrees
        
        Returns:
            Dictionary with node and edge routing information
        """
        node_routing = self.node_scorer(node_emb, incidence, node_degrees)
        edge_routing = self.edge_scorer(edge_emb, incidence)

        return {
            "node": node_routing,
            "edge": edge_routing,
        }


def selective_alignment_loss(
    z: torch.Tensor,
    align_mask: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    loss_type: str = "mse",
) -> torch.Tensor:
    """
    Compute alignment loss only for high-confidence samples.
    
    Args:
        z: Representations to align
        align_mask: Binary mask indicating which samples to align
        target: Optional target representations
        loss_type: 'mse' or 'contrastive'
    
    Returns:
        Alignment loss (only for masked samples)
    """
    if align_mask.sum() == 0:
        return z.new_tensor(0.0)

    masked_z = z[align_mask.bool()]
    
    if target is not None:
        masked_target = target[align_mask.bool()]
        if loss_type == "mse":
            loss = F.mse_loss(masked_z, masked_target)
        else:
            # Contrastive alignment
            z_norm = F.normalize(masked_z, dim=-1)
            target_norm = F.normalize(masked_target, dim=-1)
            loss = 2 - 2 * (z_norm * target_norm).sum(dim=-1).mean()
    else:
        # Self-supervised alignment (e.g., across views)
        # Simplified: encourage uniform distribution
        z_norm = F.normalize(masked_z, dim=-1)
        loss = (z_norm @ z_norm.T).pow(2).mean()

    return loss
