"""
Multi-Granularity Prototype Alignment Module for Cross-Domain Hypergraph Learning.

This module implements:
1. Node-level prototype alignment using structural descriptors
2. Hyperedge-level prototype alignment
3. Prototype management (initialization, updates)

Reference: Challenge 1 Solution Spec, Part B
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class PrototypeProjector(nn.Module):
    """Projects embeddings to prototype space for alignment."""

    def __init__(
        self,
        in_dim: int,
        proj_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        h_dim = hidden_dim or max(in_dim // 2, 64)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings (batch, in_dim)
        
        Returns:
            Projected embeddings (batch, proj_dim), normalized
        """
        return F.normalize(self.net(x), dim=-1)


class PrototypeBank(nn.Module):
    """
    Manages prototype centroids for alignment.
    
    Prototypes are cluster centers computed from structural descriptors.
    Samples are aligned to their matching prototypes.
    """

    def __init__(
        self,
        num_prototypes: int,
        proj_dim: int,
        num_domains: int = 1,
        learnable: bool = True,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.proj_dim = proj_dim
        self.num_domains = num_domains
        self.learnable = learnable

        if learnable:
            # Learnable prototype embeddings
            self.prototypes = nn.Parameter(
                torch.randn(num_prototypes * num_domains, proj_dim)
            )
        else:
            self.register_buffer(
                "prototypes",
                torch.randn(num_prototypes * num_domains, proj_dim)
            )

    def init_from_descriptors(
        self,
        descriptors: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize prototypes using k-means on structural descriptors.
        
        Args:
            descriptors: Structural descriptor features (N, feat_dim)
            labels: Optional domain labels (N,)
            device: Device to use
        """
        if labels is None:
            # Single domain
            descriptors_np = descriptors.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=self.num_prototypes, random_state=42, n_init=10)
            kmeans.fit(descriptors_np)
            centroids = torch.from_numpy(kmeans.cluster_centers_).float()
            
            if device is not None:
                centroids = centroids.to(device)
            
            if self.learnable:
                self.prototypes.data[:self.num_prototypes] = F.normalize(centroids, dim=-1)
            else:
                self.prototypes[:self.num_prototypes] = F.normalize(centroids, dim=-1)
        else:
            # Multi-domain prototypes
            for d in range(self.num_domains):
                mask = labels == d
                if mask.sum() > 0:
                    domain_descs = descriptors[mask].detach().cpu().numpy()
                    kmeans = KMeans(
                        n_clusters=self.num_prototypes,
                        random_state=42 + d,
                        n_init=10
                    )
                    kmeans.fit(domain_descs)
                    centroids = torch.from_numpy(kmeans.cluster_centers_).float()
                    start_idx = d * self.num_prototypes
                    end_idx = (d + 1) * self.num_prototypes
                    if device is not None:
                        centroids = centroids.to(device)
                    if self.learnable:
                        self.prototypes.data[start_idx:end_idx] = F.normalize(centroids, dim=-1)
                    else:
                        self.prototypes[start_idx:end_idx] = F.normalize(centroids, dim=-1)

    def get_prototypes(self, domain_id: Optional[int] = None) -> torch.Tensor:
        """
        Get prototype embeddings.
        
        Args:
            domain_id: Optional domain ID for domain-specific prototypes
        
        Returns:
            Prototype embeddings
        """
        if domain_id is None or not self.learnable:
            return F.normalize(self.prototypes, dim=-1)
        
        start_idx = domain_id * self.num_prototypes
        end_idx = (domain_id + 1) * self.num_prototypes
        return F.normalize(self.prototypes[start_idx:end_idx], dim=-1)


def prototype_alignment_loss(
    z: torch.Tensor,
    proto_ids: torch.Tensor,
    proto_table: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute prototype alignment loss.
    
    Args:
        z: Projected embeddings (N, proj_dim)
        proto_ids: Prototype assignment IDs (N,)
        proto_table: Prototype embeddings (num_prototypes, proj_dim)
        mask: Optional mask for selective alignment
    
    Returns:
        Prototype alignment loss
    """
    if mask is not None:
        # Convert to bool if mask is float tensor (e.g., from confidence scoring)
        if mask.dtype == torch.float32:
            mask = mask.bool()
        z = z[mask]
        proto_ids = proto_ids[mask]

    if z.size(0) == 0:
        return z.new_tensor(0.0)

    z_norm = F.normalize(z, dim=-1)
    target = F.normalize(proto_table[proto_ids], dim=-1)
    
    # Pull toward matching prototype
    loss = (1 - (z_norm * target).sum(dim=-1)).mean()
    
    return loss


def supervised_contrastive_alignment(
    z: torch.Tensor,
    proto_ids: torch.Tensor,
    proto_table: torch.Tensor,
    temperature: float = 0.1,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Supervised contrastive alignment using prototypes as anchors.
    
    Samples with the same prototype ID are treated as positives.
    
    Args:
        z: Projected embeddings
        proto_ids: Prototype assignment IDs
        proto_table: Prototype embeddings
        temperature: Contrastive temperature
        mask: Optional mask
    
    Returns:
        Contrastive alignment loss
    """
    if mask is not None:
        z = z[mask]
        proto_ids = proto_ids[mask]

    if z.size(0) == 0:
        return z.new_tensor(0.0)

    z_norm = F.normalize(z, dim=-1)
    
    # Compute similarities to prototypes
    sim = z_norm @ F.normalize(proto_table, dim=-1).T / temperature
    
    # Positive pairs: same prototype ID
    positives = (proto_ids.unsqueeze(1) == proto_ids.unsqueeze(0)).float()
    
    # Mask out self-comparisons
    positives.fill_diagonal_(0)
    
    # Normalize positives per row
    pos_sum = positives.sum(dim=1, keepdim=True).clamp(min=1)
    pos_mask = positives / pos_sum
    
    # InfoNCE loss
    log_probs = sim.log_softmax(dim=1)
    loss = -(pos_mask * log_probs).sum(dim=1).mean()
    
    return loss


class NodePrototypeAlignment(nn.Module):
    """
    Node-level prototype alignment.
    
    Uses structural descriptors (degree, overlap, etc.) to cluster nodes
    and align their shared representations to domain-invariant prototypes.
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: int,
        num_prototypes: int = 32,
        num_domains: int = 1,
        alignment_type: str = "prototype",
        temperature: float = 0.1,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.alignment_type = alignment_type

        self.projector = PrototypeProjector(in_dim, proj_dim)
        self.proto_bank = PrototypeBank(
            num_prototypes=num_prototypes,
            proj_dim=proj_dim,
            num_domains=num_domains,
        )
        self.temperature = temperature

    def assign_to_prototypes(
        self,
        node_emb: torch.Tensor,
        descriptors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign nodes to nearest prototypes based on embeddings.
        
        Args:
            node_emb: Node embeddings
            descriptors: Structural descriptors for clustering
        
        Returns:
            proto_ids: Prototype assignments
            proto_table: Current prototype table
        """
        proj_emb = self.projector(node_emb)
        proj_norm = F.normalize(proj_emb, dim=-1)
        
        proto_table = self.proto_bank.get_prototypes()
        
        # Assign based on embedding similarity
        sim = proj_norm @ proto_table.T
        proto_ids = sim.argmax(dim=-1)
        
        return proto_ids, proto_table

    def forward(
        self,
        node_emb: torch.Tensor,
        descriptors: torch.Tensor,
        align_mask: Optional[torch.Tensor] = None,
        domain_id: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute node prototype alignment loss.
        
        Args:
            node_emb: Node embeddings
            descriptors: Structural descriptors
            align_mask: Mask for selective alignment
            domain_id: Optional domain ID
        
        Returns:
            Dictionary with alignment loss and statistics
        """
        proj_emb = self.projector(node_emb)
        proto_ids, proto_table = self.assign_to_prototypes(node_emb, descriptors)
        
        if align_mask is not None:
            loss = prototype_alignment_loss(proj_emb, proto_ids, proto_table, align_mask)
        else:
            loss = prototype_alignment_loss(proj_emb, proto_ids, proto_table)

        return {
            "loss": loss,
            "proto_ids": proto_ids,
            "num_prototypes_used": len(torch.unique(proto_ids)),
        }


class EdgePrototypeAlignment(nn.Module):
    """
    Hyperedge-level prototype alignment.
    
    Uses edge structural descriptors to create domain-invariant
    edge prototypes for alignment.
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: int,
        num_prototypes: int = 32,
        num_domains: int = 1,
        alignment_type: str = "prototype",
        temperature: float = 0.1,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.alignment_type = alignment_type

        self.projector = PrototypeProjector(in_dim, proj_dim)
        self.proto_bank = PrototypeBank(
            num_prototypes=num_prototypes,
            proj_dim=proj_dim,
            num_domains=num_domains,
        )
        self.temperature = temperature

    def assign_to_prototypes(
        self,
        edge_emb: torch.Tensor,
        descriptors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign edges to nearest prototypes.
        """
        proj_emb = self.projector(edge_emb)
        proj_norm = F.normalize(proj_emb, dim=-1)
        
        proto_table = self.proto_bank.get_prototypes()
        
        sim = proj_norm @ proto_table.T
        proto_ids = sim.argmax(dim=-1)
        
        return proto_ids, proto_table

    def forward(
        self,
        edge_emb: torch.Tensor,
        descriptors: torch.Tensor,
        align_mask: Optional[torch.Tensor] = None,
        domain_id: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute edge prototype alignment loss.
        """
        proj_emb = self.projector(edge_emb)
        proto_ids, proto_table = self.assign_to_prototypes(edge_emb, descriptors)
        
        if align_mask is not None:
            loss = prototype_alignment_loss(proj_emb, proto_ids, proto_table, align_mask)
        else:
            loss = prototype_alignment_loss(proj_emb, proto_ids, proto_table)

        return {
            "loss": loss,
            "proto_ids": proto_ids,
            "num_prototypes_used": len(torch.unique(proto_ids)),
        }


class MultiGranularityAlignment(nn.Module):
    """
    Unified multi-granularity alignment module.
    
    Aligns shared representations at both node and edge levels
    using prototype-based methods.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        proj_dim: int = 64,
        node_num_prototypes: int = 32,
        edge_num_prototypes: int = 32,
        num_domains: int = 1,
        use_node_alignment: bool = True,
        use_edge_alignment: bool = True,
    ):
        super().__init__()
        self.use_node_alignment = use_node_alignment
        self.use_edge_alignment = use_edge_alignment

        if use_node_alignment:
            self.node_aligner = NodePrototypeAlignment(
                in_dim=node_dim,
                proj_dim=proj_dim,
                num_prototypes=node_num_prototypes,
                num_domains=num_domains,
            )

        if use_edge_alignment:
            self.edge_aligner = EdgePrototypeAlignment(
                in_dim=edge_dim,
                proj_dim=proj_dim,
                num_prototypes=edge_num_prototypes,
                num_domains=num_domains,
            )

    def forward(
        self,
        node_emb: Optional[torch.Tensor] = None,
        edge_emb: Optional[torch.Tensor] = None,
        node_descriptors: Optional[torch.Tensor] = None,
        edge_descriptors: Optional[torch.Tensor] = None,
        node_align_mask: Optional[torch.Tensor] = None,
        edge_align_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-granularity alignment losses.
        
        Args:
            node_emb: Node shared embeddings
            edge_emb: Edge shared embeddings
            node_descriptors: Node structural descriptors
            edge_descriptors: Edge structural descriptors
            node_align_mask: Mask for node alignment (confidence-based)
            edge_align_mask: Mask for edge alignment
        
        Returns:
            Dictionary with alignment losses
        """
        losses = {}

        if self.use_node_alignment and node_emb is not None:
            node_result = self.node_aligner(
                node_emb,
                node_descriptors,
                node_align_mask,
            )
            losses["align_node"] = node_result["loss"]
            losses["node_proto_ids"] = node_result["proto_ids"]

        if self.use_edge_alignment and edge_emb is not None:
            edge_result = self.edge_aligner(
                edge_emb,
                edge_descriptors,
                edge_align_mask,
            )
            losses["align_edge"] = edge_result["loss"]
            losses["edge_proto_ids"] = edge_result["proto_ids"]

        return losses


def build_structural_descriptors_from_hg(
    hg: "SimpleHypergraph",  # noqa: F821
    node_emb: Optional[torch.Tensor] = None,
    edge_emb: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Build structural descriptors from hypergraph.
    
    This is a simplified version that doesn't require sklearn.
    
    Args:
        hg: SimpleHypergraph
        node_emb: Optional node embeddings for embedding-based features
        edge_emb: Optional edge embeddings
    
    Returns:
        node_descriptors, edge_descriptors
    """
    num_nodes = hg.num_nodes
    num_edges = len(hg.hyperedges)

    # Node descriptors: degree-based features
    node_degrees = torch.zeros(num_nodes)
    for i in range(num_nodes):
        for e_idx, edge in enumerate(hg.hyperedges):
            if i in edge:
                node_degrees[i] += 1

    node_feat_dim = 4
    node_descriptors = torch.zeros(num_nodes, node_feat_dim)
    max_degree = max(node_degrees.max().item(), 1.0)
    
    for i in range(num_nodes):
        # Feature 1: Normalized degree
        node_descriptors[i, 0] = node_degrees[i] / max_degree
        
        # Feature 2: Binary - has connections
        node_descriptors[i, 1] = 1.0 if node_degrees[i] > 0 else 0.0
        
        # Feature 3: Degree category (normalized)
        node_descriptors[i, 2] = min(node_degrees[i] / 5.0, 1.0)
        
        # Feature 4: Local clustering estimate
        incident_edges = [e_idx for e_idx, edge in enumerate(hg.hyperedges) if i in edge]
        if len(incident_edges) >= 2:
            node_descriptors[i, 3] = 0.5  # Simplified
        else:
            node_descriptors[i, 3] = 0.0

    # Edge descriptors
    edge_feat_dim = 4
    edge_descriptors = torch.zeros(num_edges, edge_feat_dim)
    
    for e_idx, edge in enumerate(hg.hyperedges):
        # Feature 1: Cardinality
        edge_descriptors[e_idx, 0] = len(edge) / 20.0
        
        # Feature 2: Binary - valid size
        edge_descriptors[e_idx, 1] = 1.0 if 2 <= len(edge) <= 10 else 0.5
        
        # Feature 3: Overlap with other edges
        edge_nodes = set(edge)
        overlap_count = 0
        for other_idx, other_edge in enumerate(hg.hyperedges):
            if other_idx != e_idx and edge_nodes & set(other_edge):
                overlap_count += 1
        edge_descriptors[e_idx, 2] = overlap_count / max(num_edges, 1)
        
        # Feature 4: Member degree average
        member_degrees = [node_degrees[n] for n in edge]
        avg_member_degree = sum(member_degrees) / max(len(member_degrees), 1)
        edge_descriptors[e_idx, 3] = avg_member_degree / max_degree

    return node_descriptors, edge_descriptors
