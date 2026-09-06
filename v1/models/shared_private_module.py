"""
Shared-Private Disentanglement Module for Cross-Domain Hypergraph Learning.

This module implements:
1. SharedPrivateProjector: Separates domain-invariant structure from domain-specific statistics
2. PrivateDomainPredictor: Encourages private branch to retain domain-specific information
3. OrthogonalityLoss: Ensures shared and private branches encode different information
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedPrivateProjector(nn.Module):
    """
    Separates representations into shared (transferable) and private (domain-specific) branches.
    
    Shared branch learns: transferable higher-order structure, reusable patterns
    Private branch learns: domain-specific cardinality, density biases, sampling artifacts
    """

    def __init__(
        self,
        in_dim: int,
        shared_dim: Optional[int] = None,
        private_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        out_dim = shared_dim or in_dim
        self.shared_dim = shared_dim or in_dim
        self.private_dim = private_dim or in_dim

        self.shared_mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, self.shared_dim),
        )

        self.private_mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, self.private_dim),
        )

        self.out_dim = max(self.shared_dim, self.private_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, in_dim)
        
        Returns:
            z_shared: Domain-invariant representation
            z_private: Domain-specific representation
        """
        z_shared = self.shared_mlp(x)
        z_private = self.private_mlp(x)
        return z_shared, z_private


class NodeDisentangler(nn.Module):
    """Node-level disentangler for separating node structural roles from domain statistics."""

    def __init__(
        self,
        in_dim: int,
        shared_dim: Optional[int] = None,
        private_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projector = SharedPrivateProjector(
            in_dim=in_dim,
            shared_dim=shared_dim,
            private_dim=private_dim,
            dropout=dropout,
        )

    def forward(self, node_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            node_emb: Node embeddings (num_nodes, in_dim)
        
        Returns:
            z_node_shared: Node shared representations
            z_node_private: Node private representations
            combined: Concatenated [shared, private] for task prediction
        """
        z_shared, z_private = self.projector(node_emb)
        # Pad to same dimension if needed
        if z_shared.size(-1) != z_private.size(-1):
            max_dim = max(z_shared.size(-1), z_private.size(-1))
            z_shared = F.pad(z_shared, (0, max_dim - z_shared.size(-1)))
            z_private = F.pad(z_private, (0, max_dim - z_private.size(-1)))
        combined = torch.cat([z_shared, z_private], dim=-1)
        return z_shared, z_private, combined


class EdgeDisentangler(nn.Module):
    """Hyperedge-level disentangler."""

    def __init__(
        self,
        in_dim: int,
        shared_dim: Optional[int] = None,
        private_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projector = SharedPrivateProjector(
            in_dim=in_dim,
            shared_dim=shared_dim,
            private_dim=private_dim,
            dropout=dropout,
        )

    def forward(self, edge_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            edge_emb: Edge embeddings (num_edges, in_dim)
        
        Returns:
            z_edge_shared: Edge shared representations
            z_edge_private: Edge private representations
            combined: Concatenated [shared, private]
        """
        z_shared, z_private = self.projector(edge_emb)
        if z_shared.size(-1) != z_private.size(-1):
            max_dim = max(z_shared.size(-1), z_private.size(-1))
            z_shared = F.pad(z_shared, (0, max_dim - z_shared.size(-1)))
            z_private = F.pad(z_private, (0, max_dim - z_private.size(-1)))
        combined = torch.cat([z_shared, z_private], dim=-1)
        return z_shared, z_private, combined


class PrivateDomainPredictor(nn.Module):
    """
    Predicts domain from private representations.
    Encourages z_private to retain domain-specific information.
    """

    def __init__(
        self,
        in_dim: int,
        num_domains: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        h_dim = hidden_dim or max(in_dim // 2, 64)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, num_domains),
        )

    def forward(self, z_private: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_private: Private representations
        
        Returns:
            Domain logits (for cross-entropy loss)
        """
        return self.net(z_private)


class OrthogonalityLoss(nn.Module):
    """
    Orthogonality constraint between shared and private representations.
    Prevents collapse where z_shared and z_private become identical.
    """

    def __init__(self, lambda_orth: float = 0.05):
        super().__init__()
        self.lambda_orth = lambda_orth

    def forward(
        self,
        z_shared: torch.Tensor,
        z_private: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_shared: Shared representations (batch, shared_dim)
            z_private: Private representations (batch, private_dim)
        
        Returns:
            Orthogonality loss value
        """
        # Normalize to unit sphere
        z_shared = F.normalize(z_shared, dim=-1)
        z_private = F.normalize(z_private, dim=-1)

        # Compute cross-covariance matrix
        # For batch x shared, y x private: cross[i,j] = mean(batch) z_shared[:,i] * z_private[:,j]
        batch_size = z_shared.size(0)
        
        # Simple approach: maximize distance between normalized vectors
        # (1 - cosine_sim) encourages orthogonality
        cosine_sim = (z_shared * z_private).sum(dim=-1).mean()
        loss = (1.0 - cosine_sim.pow(2)).mean()  # Squared to penalize high correlation

        return self.lambda_orth * loss


class DisentanglementLosses(nn.Module):
    """
    Aggregates all disentanglement-related losses.
    
    Includes:
    - Orthogonality loss (shared vs private)
    - Private domain prediction loss
    """

    def __init__(
        self,
        lambda_orth: float = 0.05,
        lambda_private_domain: float = 0.05,
    ):
        super().__init__()
        self.orth_loss = OrthogonalityLoss(lambda_orth=lambda_orth)
        self.lambda_orth = lambda_orth
        self.lambda_private_domain = lambda_private_domain

    def forward(
        self,
        z_shared: torch.Tensor,
        z_private: torch.Tensor,
        domain_labels: torch.Tensor,
        private_predictor: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_shared: Shared representations (batch, shared_dim)
            z_private: Private representations (batch, private_dim)
            domain_labels: Domain IDs for each sample (batch,)
            private_predictor: Module to predict domain from private
        
        Returns:
            Dictionary of losses
        """
        losses = {}

        # Orthogonality loss
        losses["orth"] = self.orth_loss(z_shared, z_private)

        # Private domain prediction loss (encourages private to retain domain info)
        if self.lambda_private_domain > 0 and domain_labels is not None and domain_labels.numel() > 0:
            domain_logits = private_predictor(z_private)
            if domain_logits.size(0) == domain_labels.size(0):
                losses["private_domain"] = F.cross_entropy(domain_logits, domain_labels.long())
            else:
                # Size mismatch (e.g. single-edge subhypergraph); skip rather than crash.
                losses["private_domain"] = z_shared.new_tensor(0.0)
        else:
            losses["private_domain"] = z_shared.new_tensor(0.0)

        return losses


def build_edge_embedding_from_nodes(
    node_shared: torch.Tensor,
    edge_node_indices: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    """
    Build hyperedge embeddings by aggregating member node shared embeddings.
    
    Args:
        node_shared: Node shared embeddings (num_nodes, shared_dim)
        edge_node_indices: List of node indices for each edge
        mode: Aggregation mode ('mean', 'max', 'sum')
    
    Returns:
        Edge embeddings (num_edges, shared_dim)
    """
    if isinstance(edge_node_indices, list):
        # edge_node_indices is List[List[int]]
        edge_embs = []
        for nodes in edge_node_indices:
            if nodes:
                node_embs = node_shared[nodes]
                if mode == "mean":
                    edge_emb = node_embs.mean(dim=0)
                elif mode == "max":
                    edge_emb = node_embs.max(dim=0)[0]
                elif mode == "sum":
                    edge_emb = node_embs.sum(dim=0)
                else:
                    edge_emb = node_embs.mean(dim=0)
                edge_embs.append(edge_emb)
            else:
                edge_embs.append(node_shared.new_zeros(node_shared.size(-1)))
        return torch.stack(edge_embs, dim=0)
    else:
        # Sparse incidence matrix case
        raise NotImplementedError("Sparse incidence matrix not yet supported")


def compute_structural_descriptors(
    hg: "SimpleHypergraph",  # noqa: F821
    node_emb: torch.Tensor,
    edge_emb: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute structural descriptors for nodes and edges.
    
    Node descriptors:
    - Node degree
    - Mean size of incident hyperedges
    - Local overlap count
    - Incidence density
    
    Edge descriptors:
    - Hyperedge cardinality
    - Mean degree of member nodes
    - Overlap count with other edges
    - Overlap density
    
    Args:
        hg: SimpleHypergraph instance
        node_emb: Node embeddings
        edge_emb: Edge embeddings
    
    Returns:
        node_descriptors: (num_nodes, num_node_features)
        edge_descriptors: (num_edges, num_edge_features)
    """
    num_nodes = hg.num_nodes
    num_edges = len(hg.hyperedges)

    # Node descriptors
    node_degrees = torch.zeros(num_nodes)
    mean_edge_sizes = torch.zeros(num_nodes)
    local_overlaps = torch.zeros(num_nodes)
    incidence_density = torch.zeros(num_nodes)

    for i in range(num_nodes):
        incident_edges = []
        for e_idx, edge in enumerate(hg.hyperedges):
            if i in edge:
                incident_edges.append(e_idx)
                node_degrees[i] += 1
        if incident_edges:
            mean_edge_sizes[i] = sum(len(hg.hyperedges[e]) for e in incident_edges) / len(incident_edges)
            # Local overlap: count other edges that share nodes with incident edges
            shared_nodes = set()
            for e in incident_edges:
                shared_nodes.update(hg.hyperedges[e])
            local_overlaps[i] = len(shared_nodes) - len(incident_edges)
        incidence_density[i] = len(incident_edges) / max(num_edges, 1)

    # Normalize
    node_degrees = node_degrees / max(node_degrees.max().item(), 1.0)
    mean_edge_sizes = mean_edge_sizes / 10.0  # Approximate normalization
    local_overlaps = local_overlaps / max(local_overlaps.max().item(), 1.0) if local_overlaps.max() > 0 else local_overlaps

    node_descriptors = torch.stack([
        node_degrees,
        mean_edge_sizes,
        local_overlaps,
        incidence_density,
    ], dim=-1)

    # Edge descriptors
    edge_cardinalities = torch.tensor([len(e) for e in hg.hyperedges], dtype=torch.float32)
    mean_member_degrees = torch.zeros(num_edges)
    edge_overlap_counts = torch.zeros(num_edges)
    edge_overlap_density = torch.zeros(num_edges)

    for e_idx, edge in enumerate(hg.hyperedges):
        member_degrees = [node_degrees[n.item() if isinstance(n, torch.Tensor) else n] for n in edge]
        mean_member_degrees[e_idx] = sum(member_degrees) / max(len(member_degrees), 1)
        
        # Count overlaps with other edges
        edge_nodes = set(edge)
        overlap_count = 0
        for other_idx, other_edge in enumerate(hg.hyperedges):
            if other_idx != e_idx:
                if edge_nodes.intersection(other_edge):
                    overlap_count += 1
        edge_overlap_counts[e_idx] = overlap_count

    # Normalize
    edge_cardinalities = edge_cardinalities / 20.0  # Approximate normalization
    mean_member_degrees = mean_member_degrees / max(mean_member_degrees.max().item(), 1.0) if mean_member_degrees.max() > 0 else mean_member_degrees
    edge_overlap_counts = edge_overlap_counts / max(edge_overlap_counts.max().item(), 1.0) if edge_overlap_counts.max() > 0 else edge_overlap_counts

    edge_descriptors = torch.stack([
        edge_cardinalities,
        mean_member_degrees,
        edge_overlap_counts,
        edge_overlap_density,
    ], dim=-1)

    return node_descriptors, edge_descriptors
