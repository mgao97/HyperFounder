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

        # Overlap score: for each node i, sum over co-member nodes u of
        # C(co[i,u], 2) where co[i,u] = #edges containing both i and u. This is
        # exactly the pairwise edge-intersection semantics of the original
        # O(N*k^2) Python loop, but computed WITHOUT any Python iteration over
        # node pairs and WITHOUT dense materialisation.
        #
        # We build the node-node co-membership as a sparse COO tensor by
        # enumerating unordered node pairs per edge (each edge of size k_e
        # contributes C(k_e,2) pairs; small for real hypergraphs). A coalesced
        # sparse tensor then sums duplicate (i,u) entries into co[i,u]. This
        # avoids torch.sparse.mm (which is unstable on CPU and OOM-prone on
        # GPU for dense-ish inputs) and the original set-intersection loop.
        # Hyperedges larger than MAX_PAIR_MEMBERS are skipped — their
        # contribution to pairwise overlap is negligible and this bounds work.
        overlap_score = torch.zeros(num_nodes, device=device)
        if col.numel() > 0:
            MAX_PAIR_MEMBERS = 2048
            e_order = torch.argsort(col, stable=True)
            col_e, row_e = col[e_order], row[e_order]
            uniq_e, e_counts = torch.unique_consecutive(col_e, return_counts=True)
            # e_start[j] / e_start[j+1] delimit the member range of edge uniq_e[j]
            e_start = torch.cat([
                torch.zeros(1, dtype=torch.long, device=device), e_counts.cumsum(0)
            ])
            src_list, dst_list = [], []
            triu_cache = {}
            for j in range(uniq_e.numel()):
                st, en = int(e_start[j]), int(e_start[j + 1])
                k = en - st
                if k < 2 or k > MAX_PAIR_MEMBERS:
                    continue
                m = row_e[st:en]
                triu = triu_cache.get(k)
                if triu is None:
                    triu = torch.triu(
                        torch.ones(k, k, dtype=torch.bool, device=device), diagonal=1
                    )
                    triu_cache[k] = triu
                pairs_i = m.unsqueeze(0).expand(k, k)[triu]
                pairs_j = m.unsqueeze(1).expand(k, k)[triu]
                src_list.append(pairs_i)
                dst_list.append(pairs_j)
                src_list.append(pairs_j)  # symmetric: co[i,u] == co[u,i]
                dst_list.append(pairs_i)
            if src_list:
                src = torch.cat(src_list)
                dst = torch.cat(dst_list)
                co = torch.sparse_coo_tensor(
                    torch.stack([src, dst]),
                    torch.ones(src.numel(), device=device),
                    (num_nodes, num_nodes),
                ).coalesce()
                cidx = co.indices()
                cval = co.values().float()
                comb = cval * (cval - 1.0) / 2.0  # C(v, 2)
                raw = torch.zeros(num_nodes, device=device).index_add(0, cidx[0], comb)
                # Diagonal term: node i participates in its own edge intersections
                # (|M(e1) ∩ M(e2)| includes i itself), equivalent to co[i,i] = deg_i.
                deg_i = node_degrees.long().clamp(min=0)
                raw = raw + (deg_i * (deg_i - 1) // 2).float()  # C(deg_i, 2)
                denom = (deg_i * (deg_i - 1) // 2).clamp(min=1).float()
                overlap_score = (raw / denom).clamp(0, 1)

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

        # Edge -> member count (cardinality) from sparse COO. Length is
        # num_edges (not max edge id) so empty/sparse edge ids can't overflow.
        edge_card = torch.zeros(num_edges, dtype=torch.long, device=device)
        if col.numel() > 0:
            edge_card = edge_card.index_add(
                0, col.clamp(max=num_edges - 1), torch.ones(col.numel(), dtype=torch.long, device=device)
            )

        # Overlap score: for each edge e, count how many *other* edges share at
        # least one node with e (de-duplicated), normalised by (num_edges - 1).
        # This is exactly the semantics of the original O(E^2) loop
        # (`for e_idx, for other_idx: if share a node -> overlap_count += 1`)
        # but computed via a sparse shared-edge matrix (no Python loop, no set
        # ops, no dense materialisation). Edges sharing >=1 node get clamped 0/1.
        overlap_share = torch.zeros(num_edges, device=device)
        if col.numel() > 0:
            n_order = torch.argsort(row, stable=True)
            row_n, col_n = row[n_order], col[n_order]
            uniq_n, n_counts = torch.unique_consecutive(row_n, return_counts=True)
            n_start = torch.cat([
                torch.zeros(1, dtype=torch.long, device=device), n_counts.cumsum(0)
            ])
            src_list, dst_list = [], []
            comb_cache = {}
            for j in range(uniq_n.numel()):
                st, en = int(n_start[j]), int(n_start[j + 1])
                k = en - st
                if k < 2:
                    continue
                m = col_n[st:en]  # incident edges of this node
                triu = comb_cache.get(k)
                if triu is None:
                    triu = torch.triu(
                        torch.ones(k, k, dtype=torch.bool, device=device), diagonal=1
                    )
                    comb_cache[k] = triu
                a = m.unsqueeze(0).expand(k, k)[triu]
                b = m.unsqueeze(1).expand(k, k)[triu]
                src_list += [a, b]
                dst_list += [b, a]
            if src_list:
                src = torch.cat(src_list)
                dst = torch.cat(dst_list)
                share = torch.sparse_coo_tensor(
                    torch.stack([src, dst]),
                    torch.ones(src.numel(), device=device),
                    (num_edges, num_edges),
                ).coalesce()
                sv = share.values().clamp(max=1).float()  # 0/1: shares a node or not
                overlap_share = torch.zeros(num_edges, device=device).index_add(
                    0, share.indices()[0], sv
                )
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
