from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Optional, Tuple

from models.shared_private_module import (
    SharedPrivateProjector,
    NodeDisentangler,
    EdgeDisentangler,
    PrivateDomainPredictor,
    DisentanglementLosses,
    build_edge_embedding_from_nodes,
    compute_structural_descriptors,
)
from models.domain_alignment import (
    NodePrototypeAlignment,
    EdgePrototypeAlignment,
    MultiGranularityAlignment,
    build_structural_descriptors_from_hg,
)
from models.domain_confidence import (
    ConfidenceRouter,
    selective_alignment_loss,
)


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


class MotifCounter(nn.Module):
    """Motif counting head for hypergraph structure prediction."""

    def __init__(self, hidden_dim: int, num_motif_types: int = 8):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_motif_types),
        )

    def forward(self, node_emb: torch.Tensor, edge_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([node_emb.mean(dim=0), edge_emb.mean(dim=0)], dim=-1)
        return self.classifier(combined)


class CommunityPrototypeHead(nn.Module):
    """Community prototype alignment head."""

    def __init__(self, hidden_dim: int, num_prototypes: int = 8):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prototype_embeddings = nn.Parameter(torch.randn(num_prototypes, hidden_dim))
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        proj_emb = self.projector(emb)
        sim = proj_emb @ self.prototype_embeddings.T
        return sim, self.prototype_embeddings


class StructureAlignmentHead(nn.Module):
    """Structure-aware alignment head for multi-granularity consistency."""

    def __init__(self, hidden_dim: int, alignment_dim: int = 64):
        super().__init__()
        self.node_structure_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, alignment_dim),
        )
        self.edge_structure_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, alignment_dim),
        )
        self.motif_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, alignment_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(alignment_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, alignment_dim),
        )

    def encode_node_structure(self, node_emb: torch.Tensor, edge_emb: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        dense_inc = incidence.to_dense() if incidence.is_sparse else incidence
        num_nodes = node_emb.size(0)
        node_context = torch.zeros_like(node_emb)
        for i in range(num_nodes):
            incident_edges = dense_inc[i].nonzero(as_tuple=True)[0]
            if incident_edges.numel() > 0:
                node_context[i] = edge_emb[incident_edges].mean(dim=0)
        combined = torch.cat([node_emb, node_context], dim=-1)
        return self.node_structure_encoder(combined)

    def encode_edge_structure(self, node_emb: torch.Tensor, edge_emb: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        dense_inc = incidence.to_dense() if incidence.is_sparse else incidence
        num_edges = edge_emb.size(0)
        edge_context = torch.zeros_like(edge_emb)
        for i in range(num_edges):
            members = dense_inc[:, i].nonzero(as_tuple=True)[0]
            if members.numel() > 0:
                edge_context[i] = node_emb[members].mean(dim=0)
        combined = torch.cat([edge_emb, edge_context], dim=-1)
        return self.edge_structure_encoder(combined)

    def forward(self, node_emb: torch.Tensor, edge_emb: torch.Tensor, motif_emb: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        node_struct = self.encode_node_structure(node_emb, edge_emb, incidence)
        edge_struct = self.encode_edge_structure(node_emb, edge_emb, incidence)
        motif_pooled = motif_emb.mean(dim=0) if motif_emb.numel() > 0 else torch.zeros_like(node_emb[:1]).squeeze(0)
        motif_struct = self.motif_encoder(torch.cat([motif_pooled, motif_pooled], dim=-1))
        combined = torch.cat([node_struct.mean(dim=0), edge_struct.mean(dim=0), motif_struct], dim=-1)
        return self.projector(combined)


class TaskRouter(nn.Module):
    """Task routing based on subgraph structure characteristics."""

    def __init__(
        self,
        hidden_dim: int,
        num_tasks: int = 6,
        small_threshold: int = 32,
        medium_threshold: int = 128,
        feature_dim: int = 5,
    ):
        super().__init__()
        self.small_threshold = small_threshold
        self.medium_threshold = medium_threshold
        self.feature_dim = feature_dim
        self.structure_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
        )
        self.router = nn.Linear(32, num_tasks)

    def compute_structure_features(
        self,
        num_nodes: int,
        num_edges: int,
        incidence: torch.Tensor,
    ) -> torch.Tensor:
        """Compute structure-based features for routing decision."""
        device = incidence.device if hasattr(incidence, 'device') and incidence.device.type != 'meta' else 'cpu'
        
        # Basic structural features
        basic_features = [
            num_nodes / 500.0,  # Normalized node count
            num_edges / 100.0,  # Normalized edge count
            1.0,  # Placeholder
            1.0,  # Placeholder
            0.1,  # Placeholder
        ]
        
        try:
            dense_inc = incidence.to_dense() if incidence.is_sparse else incidence
            if dense_inc.numel() > 0:
                node_degree = dense_inc.sum(dim=1).float()
                edge_cardinality = dense_inc.sum(dim=0).float()
                basic_features[2] = min(node_degree.mean().item() / 10.0, 10.0) if node_degree.numel() > 0 else 1.0
                basic_features[3] = min(edge_cardinality.mean().item() / 5.0, 10.0) if edge_cardinality.numel() > 0 else 1.0
                basic_features[4] = min((dense_inc > 0).float().mean().item(), 1.0)
        except Exception:
            pass
        
        features = torch.tensor(basic_features, dtype=torch.float32, device=device)
        return features

    def forward(
        self,
        num_nodes: int,
        num_edges: int,
        incidence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns task weights for each of the 6 tasks:
        [masked_node, hyperedge_recon, contrastive, size_pred, motif, community]
        """
        features = self.compute_structure_features(num_nodes, num_edges, incidence)
        logits = self.router(self.structure_encoder(features.unsqueeze(0)))
        weights = torch.softmax(logits, dim=-1).squeeze(0)
        return weights

    def get_task_weights(
        self,
        num_nodes: int,
        num_edges: int,
        incidence: torch.Tensor,
    ) -> List[float]:
        """Get soft task weights based on subgraph structure."""
        with torch.no_grad():
            weights = self.forward(num_nodes, num_edges, incidence)
        return weights.cpu().tolist()


class TaskHeadsNegSam(nn.Module):
    """
    Enhanced task heads with disentanglement and multi-granularity alignment.
    
    Integrates:
    - Shared-private disentanglement for nodes and edges
    - Confidence-based selective alignment
    - Multi-granularity prototype alignment
    - Orthogonality constraints
    """

    def __init__(
        self,
        hidden_dim: int,
        input_dim: int,
        num_domains: int,
        projection_dim: int = 64,
        num_motif_types: int = 8,
        num_prototypes: int = 8,
        # Domain alignment settings
        shared_dim: Optional[int] = None,
        private_dim: Optional[int] = None,
        lambda_orth: float = 0.05,
        lambda_private_domain: float = 0.05,
        lambda_align: float = 0.1,
        use_confidence_routing: bool = True,
        use_node_alignment: bool = True,
        use_edge_alignment: bool = True,
        tau_node_align: float = 0.6,
        tau_edge_align: float = 0.65,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains
        self.use_confidence_routing = use_confidence_routing
        self.lambda_orth = lambda_orth
        self.lambda_private_domain = lambda_private_domain
        self.lambda_align = lambda_align

        # Basic task heads
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

        # Motif/Community tasks
        self.motif_counter = MotifCounter(hidden_dim, num_motif_types)
        self.community_prototype = CommunityPrototypeHead(hidden_dim, num_prototypes)
        self.structure_alignment = StructureAlignmentHead(hidden_dim, projection_dim)
        self.task_router = TaskRouter(hidden_dim)

        # === NEW: Disentanglement Modules ===
        self.use_disentanglement = True
        s_dim = shared_dim or hidden_dim
        p_dim = private_dim or hidden_dim

        self.node_disentangler = NodeDisentangler(
            in_dim=hidden_dim,
            shared_dim=s_dim,
            private_dim=p_dim,
        )
        self.edge_disentangler = EdgeDisentangler(
            in_dim=hidden_dim,
            shared_dim=s_dim,
            private_dim=p_dim,
        )

        # Private domain predictor (encourages private to retain domain info)
        self.private_domain_predictor = PrivateDomainPredictor(
            in_dim=p_dim,
            num_domains=num_domains,
        )

        # Disentanglement losses aggregator
        self.disentanglement_losses = DisentanglementLosses(
            lambda_orth=lambda_orth,
            lambda_private_domain=lambda_private_domain,
        )

        # === NEW: Confidence-based Selective Alignment ===
        if use_confidence_routing:
            self.confidence_router = ConfidenceRouter(
                hidden_dim=hidden_dim,
                tau_node_align=tau_node_align,
                tau_edge_align=tau_edge_align,
            )

        # === NEW: Multi-granularity Prototype Alignment ===
        if use_node_alignment or use_edge_alignment:
            self.multi_granularity_aligner = MultiGranularityAlignment(
                node_dim=s_dim,
                edge_dim=s_dim,
                proj_dim=projection_dim,
                node_num_prototypes=num_prototypes,
                edge_num_prototypes=num_prototypes,
                num_domains=num_domains,
                use_node_alignment=use_node_alignment,
                use_edge_alignment=use_edge_alignment,
            )

        # === NEW: Homoscedastic Uncertainty Weighting (Kendall et al., CVPR 2018)
        # Replaces hand-tuned loss_weights with learned per-task precision.
        # For each task i we store a learnable log_sigma_i, initialized to 0.
        # The effective weight becomes: loss_i / (2 * exp(2 * log_sigma_i)) + log_sigma_i.
        uncertainty_task_names = [
            "masked_node", "hyperedge_recon", "contrastive", "size_pred",
            "domain_align", "membership_contrast", "motif", "community",
            "structure_align", "structure_discrimination",
            "orth_node", "orth_edge", "private_domain_node", "private_domain_edge",
        ]
        self.loss_log_sigmas = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            for name in uncertainty_task_names
        })

    def compute_disentanglement_losses(
        self,
        z_node_shared: torch.Tensor,
        z_node_private: torch.Tensor,
        z_edge_shared: torch.Tensor,
        z_edge_private: torch.Tensor,
        domain_labels: torch.Tensor,
        edge_domain_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute disentanglement losses for nodes and edges.
        
        Returns:
            Dictionary with orthogonality and private-domain losses
        """
        losses = {}

        # Node disentanglement
        node_losses = self.disentanglement_losses(
            z_node_shared,
            z_node_private,
            domain_labels,
            self.private_domain_predictor,
        )
        losses["orth_node"] = node_losses["orth"]
        losses["private_domain_node"] = node_losses["private_domain"]

        # Edge disentanglement: use edge_domain_labels if provided, otherwise repeat to match.
        if edge_domain_labels is None:
            num_edges = z_edge_shared.size(0)
            if num_edges == 0:
                edge_domain_labels = domain_labels[:0]
            elif domain_labels.numel() == 0:
                edge_domain_labels = torch.zeros((num_edges,), dtype=torch.long, device=domain_labels.device)
            else:
                # Repeat domain_labels to cover all edges (handles num_edges > num_nodes too).
                repeats = (num_edges + domain_labels.numel() - 1) // domain_labels.numel()
                edge_domain_labels = domain_labels.repeat(repeats)[:num_edges]
        
        edge_losses = self.disentanglement_losses(
            z_edge_shared,
            z_edge_private,
            edge_domain_labels,
            self.private_domain_predictor,
        )
        losses["orth_edge"] = edge_losses["orth"]
        losses["private_domain_edge"] = edge_losses["private_domain"]

        return losses

    def compute_alignment_losses(
        self,
        node_emb: torch.Tensor,
        edge_emb: torch.Tensor,
        node_descriptors: Optional[torch.Tensor] = None,
        edge_descriptors: Optional[torch.Tensor] = None,
        incidence: Optional[torch.Tensor] = None,
        node_align_mask: Optional[torch.Tensor] = None,
        edge_align_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-granularity alignment losses.
        
        Returns:
            Dictionary with alignment losses
        """
        losses = {}

        if not hasattr(self, 'multi_granularity_aligner'):
            return losses

        # Get shared representations from disentanglers
        z_node_shared, _, _ = self.node_disentangler(node_emb)
        z_edge_shared, _, _ = self.edge_disentangler(edge_emb)

        # Build structural descriptors if not provided
        if node_descriptors is None:
            node_descriptors = torch.randn(node_emb.size(0), 4, device=node_emb.device)
        if edge_descriptors is None:
            edge_descriptors = torch.randn(edge_emb.size(0), 4, device=edge_emb.device)

        align_losses = self.multi_granularity_aligner(
            node_emb=z_node_shared,
            edge_emb=z_edge_shared,
            node_descriptors=node_descriptors,
            edge_descriptors=edge_descriptors,
            node_align_mask=node_align_mask,
            edge_align_mask=edge_align_mask,
        )

        # Only keep scalar (loss) tensors; metadata like proto_ids would otherwise
        # leak multi-element tensors into the loss dict and break .item() later.
        for key, val in align_losses.items():
            if isinstance(val, torch.Tensor):
                if val.numel() != 1:
                    continue  # metadata, not a loss
                losses[f"align_{key}"] = val * self.lambda_align
            else:
                losses[key] = val

        return losses

    def get_confidence_routing(
        self,
        node_emb: torch.Tensor,
        edge_emb: torch.Tensor,
        incidence: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get confidence-based routing masks for selective alignment.
        """
        if not hasattr(self, 'confidence_router'):
            return {
                "node": {"align_mask": None, "confidence": None},
                "edge": {"align_mask": None, "confidence": None},
            }

        return self.confidence_router(node_emb, edge_emb, incidence)
