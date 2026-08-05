from __future__ import annotations

from typing import Dict, List, Set, Optional

import torch
import torch.nn.functional as F

from models.encoder import UnifiedHypergraphEncoder
from models.heads_neg_sam import TaskHeadsNegSam
from models.negative_sampling_neg_sam import (
    HyperedgeNegativeBatch,
    MembershipNegativeBatch,
    sample_hyperedge_negatives,
    sample_membership_negatives,
)
from utils.hypergraph import SimpleHypergraph
from utils.sampling import augment_hypergraph


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_tensor(0.0)


def _cross_view_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    if z1.numel() == 0 or z2.numel() == 0:
        return _zero(z1 if z1.numel() else z2)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    sim = z1 @ z2.transpose(0, 1) / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.transpose(0, 1), labels))


def _pool_candidate_hyperedges(node_emb: torch.Tensor, neg_edge_node_lists: List[List[int]]) -> torch.Tensor:
    if not neg_edge_node_lists:
        return node_emb.new_zeros((0, node_emb.size(-1)))
    pooled = []
    for nodes in neg_edge_node_lists:
        if nodes:
            pooled.append(node_emb[nodes].mean(dim=0))
        else:
            pooled.append(node_emb.new_zeros(node_emb.size(-1)))
    return torch.stack(pooled, dim=0)


def compute_hyperedge_discrimination_loss(
    node_emb: torch.Tensor,
    edge_emb: torch.Tensor,
    neg_batch: HyperedgeNegativeBatch,
    heads: TaskHeadsNegSam,
) -> torch.Tensor:
    if neg_batch.pos_edge_indices.numel() == 0 or not neg_batch.neg_edge_node_lists:
        return _zero(node_emb)
    pos_edge_emb = edge_emb[neg_batch.pos_edge_indices.to(edge_emb.device)]
    neg_edge_emb = _pool_candidate_hyperedges(node_emb, neg_batch.neg_edge_node_lists)
    pos_score = heads.hyperedge_scorer(pos_edge_emb).squeeze(-1)
    neg_score = heads.hyperedge_scorer(neg_edge_emb).squeeze(-1)
    return -F.logsigmoid(pos_score - neg_score).mean()


def compute_membership_contrast_loss(
    node_emb: torch.Tensor,
    edge_emb: torch.Tensor,
    mem_batch: MembershipNegativeBatch,
    heads: TaskHeadsNegSam,
) -> torch.Tensor:
    if mem_batch.pos_pairs.numel() == 0 or mem_batch.neg_pairs.numel() == 0:
        return _zero(node_emb)
    pos_pairs = mem_batch.pos_pairs.to(node_emb.device)
    neg_pairs = mem_batch.neg_pairs.to(node_emb.device)
    pos_score = heads.membership_scorer(
        node_emb[pos_pairs[:, 0]],
        edge_emb[pos_pairs[:, 1]],
    ).squeeze(-1)
    neg_score = heads.membership_scorer(
        node_emb[neg_pairs[:, 0]],
        edge_emb[neg_pairs[:, 1]],
    ).squeeze(-1)
    return -F.logsigmoid(pos_score - neg_score).mean()


def compute_subgraph_discrimination_loss(batch_embeddings: torch.Tensor, weak_mask: torch.Tensor, heads: TaskHeadsNegSam) -> torch.Tensor:
    if batch_embeddings.numel() == 0 or weak_mask.numel() == 0:
        return _zero(batch_embeddings if batch_embeddings.numel() else weak_mask.float())
    labels = (~weak_mask).float().to(batch_embeddings.device)
    logits = heads.subgraph_scorer(batch_embeddings).squeeze(-1)
    return F.binary_cross_entropy_with_logits(logits, labels)


def compute_motif_classification_loss(
    node_emb: torch.Tensor,
    edge_emb: torch.Tensor,
    motif_emb: torch.Tensor,
    motif_labels: Optional[torch.Tensor],
    heads: TaskHeadsNegSam,
    num_motif_types: int = 8,
) -> torch.Tensor:
    """Motif classification loss for hypergraph structure prediction."""
    if motif_emb.numel() == 0 or motif_labels is None:
        return _zero(node_emb)
    
    pred = heads.motif_counter(node_emb, edge_emb)
    target = motif_labels.to(node_emb.device) if motif_labels.numel() > 0 else torch.zeros(num_motif_types, device=node_emb.device, dtype=torch.long).fill_(0)
    return torch.nan_to_num(F.cross_entropy(pred.unsqueeze(0), target[:1].unsqueeze(0)), nan=0.0, posinf=0.0, neginf=0.0)


def compute_community_alignment_loss(
    node_emb: torch.Tensor,
    community_emb: torch.Tensor,
    heads: TaskHeadsNegSam,
    num_prototypes: int = 8,
) -> torch.Tensor:
    """Community prototype alignment loss."""
    if community_emb.numel() == 0:
        return _zero(node_emb)
    
    sim_matrix, prototypes = heads.community_prototype(community_emb)
    labels = torch.arange(min(sim_matrix.size(0), num_prototypes), device=sim_matrix.device)
    
    if labels.numel() == 0:
        return _zero(node_emb)
    
    return torch.nan_to_num(F.cross_entropy(sim_matrix[:len(labels)], labels), nan=0.0, posinf=0.0, neginf=0.0)


def compute_structure_alignment_loss(
    node_emb_1: torch.Tensor,
    edge_emb_1: torch.Tensor,
    node_emb_2: torch.Tensor,
    edge_emb_2: torch.Tensor,
    motif_emb_1: torch.Tensor,
    motif_emb_2: torch.Tensor,
    incidence: torch.Tensor,
    heads: TaskHeadsNegSam,
) -> torch.Tensor:
    """Structure-aware alignment loss for multi-granularity consistency."""
    if node_emb_1.numel() == 0 or node_emb_2.numel() == 0:
        return _zero(node_emb_1 if node_emb_1.numel() else node_emb_2)
    
    struct_emb_1 = heads.structure_alignment(node_emb_1, edge_emb_1, motif_emb_1, incidence)
    struct_emb_2 = heads.structure_alignment(node_emb_2, edge_emb_2, motif_emb_2, incidence)
    
    struct_emb_1 = F.normalize(struct_emb_1, dim=-1)
    struct_emb_2 = F.normalize(struct_emb_2, dim=-1)
    
    loss = 2 - 2 * (struct_emb_1 * struct_emb_2).sum(dim=-1)
    return torch.nan_to_num(loss.mean(), nan=0.0, posinf=0.0, neginf=0.0)


def get_routed_task_weights(
    heads: TaskHeadsNegSam,
    num_nodes: int,
    num_edges: int,
    incidence: torch.Tensor,
    use_routing: bool = False,
    static_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Get task weights either from router or static config."""
    if use_routing:
        try:
            weights = heads.task_router.get_task_weights(num_nodes, num_edges, incidence)
            return {
                "masked_node": weights[0],
                "hyperedge_recon": weights[1],
                "contrastive": weights[2],
                "size_pred": weights[3],
                "motif": weights[4],
                "community": weights[5],
            }
        except Exception:
            pass
    
    return static_weights or {
        "masked_node": 1.0,
        "hyperedge_recon": 1.0,
        "contrastive": 1.0,
        "size_pred": 1.0,
        "motif": 0.5,
        "community": 0.5,
    }


def compute_pretraining_losses(
    encoder: UnifiedHypergraphEncoder,
    heads: TaskHeadsNegSam,
    hg: SimpleHypergraph,
    task_cache: Dict,
    config: Dict,
    device: torch.device,
    epoch: int,
    drop_tasks: Set[str] | None = None,
) -> Dict[str, torch.Tensor | Dict[str, float]]:
    del task_cache
    disabled = drop_tasks or set()
    training_cfg = config.get("training", {})
    neg_cfg = config.get("neg_sampling", {})
    use_task_routing = bool(training_cfg.get("use_task_routing", False))
    
    x = torch.nan_to_num(hg.x.to(device), nan=0.0, posinf=0.0, neginf=0.0)
    node_emb, edge_emb, graph_emb, aux = encoder(
        hg,
        x,
        motif_budget=int(training_cfg["motif_budget"]),
        motifs=[],
        communities=[],
        motif_seed=epoch,
    )
    
    losses: Dict[str, torch.Tensor] = {}
    stats: Dict[str, float] = {
        "num_hyperedge_negatives": 0.0,
        "num_membership_negatives": 0.0,
        "num_subgraph_negatives": 0.0,
        "hyperedge_negative_rejects": 0.0,
        "membership_false_negative_rejects": 0.0,
        "avg_negative_overlap": 0.0,
        "avg_membership_hop": 0.0,
        "avg_subgraph_strength_pos": 0.0,
        "avg_subgraph_strength_neg": 0.0,
    }
    
    weight_map = training_cfg["loss_weights"]
    incidence = aux["incidence"].to(device)
    motif_emb = aux.get("motif_emb", torch.tensor([]))
    community_emb = aux.get("community_emb", torch.tensor([]))
    
    # Get routed task weights if enabled
    if use_task_routing:
        weight_map = get_routed_task_weights(
            heads=heads,
            num_nodes=hg.num_nodes,
            num_edges=len(hg.hyperedges),
            incidence=incidence,
            use_routing=True,
            static_weights=weight_map,
        )

    masked_view = augment_hypergraph(
        hg,
        feature_mask_rate=float(training_cfg.get("feature_mask_rate", 0.15)),
        edge_dropout_rate=float(training_cfg.get("edge_dropout_rate", 0.2)),
        seed=epoch * 17 + 1,
        strategy="feature_masking",
    )
    masked_node_emb, masked_edge_emb, _, masked_aux = encoder(
        masked_view,
        torch.nan_to_num(masked_view.x.to(device), nan=0.0, posinf=0.0, neginf=0.0),
        motif_budget=0,
        motifs=[],
        communities=[],
        motif_seed=epoch,
    )
    feature_mask = masked_view.metadata.get("feature_mask")
    masked_nodes = (
        feature_mask.any(dim=1).to(device)
        if feature_mask is not None
        else masked_view.metadata.get("masked_nodes", torch.zeros(hg.num_nodes, dtype=torch.bool)).to(device)
    )
    min_masked_nodes = int(training_cfg.get("min_masked_nodes", 1))
    if "masked_node" in disabled or int(masked_nodes.sum().item()) < min_masked_nodes:
        losses["masked_node"] = _zero(graph_emb)
    else:
        pred = heads.masked_node_decoder(masked_node_emb[masked_nodes])
        target = x[masked_nodes]
        losses["masked_node"] = torch.nan_to_num(F.mse_loss(pred, target), nan=0.0, posinf=0.0, neginf=0.0)

    hyperedge_neg_batch = sample_hyperedge_negatives(
        hg,
        cfg=dict(neg_cfg.get("hyperedge", {})),
        rng=epoch * 97 + len(hg.hyperedges),
    )
    stats.update(hyperedge_neg_batch.meta)
    if "hyperedge_recon" in disabled or edge_emb.numel() == 0:
        losses["hyperedge_recon"] = _zero(graph_emb)
    else:
        losses["hyperedge_recon"] = torch.nan_to_num(
            compute_hyperedge_discrimination_loss(node_emb, edge_emb, hyperedge_neg_batch, heads),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    membership_neg_batch = sample_membership_negatives(
        hg,
        cfg=dict(neg_cfg.get("membership", {})),
        rng=epoch * 131 + hg.num_nodes,
    )
    stats.update({**stats, **membership_neg_batch.meta})
    if "membership_contrast" in disabled:
        losses["membership_contrast"] = _zero(graph_emb)
    else:
        losses["membership_contrast"] = torch.nan_to_num(
            compute_membership_contrast_loss(node_emb, edge_emb, membership_neg_batch, heads),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    aug_view = augment_hypergraph(
        hg,
        feature_mask_rate=float(training_cfg.get("feature_mask_rate", 0.15)),
        edge_dropout_rate=float(training_cfg.get("edge_dropout_rate", 0.2)),
        seed=epoch * 17 + 2,
        strategy=str(training_cfg.get("contrastive_strategy", "node_dropping")),
    )
    contrastive_node_emb, contrastive_edge_emb, _, contrastive_aux = encoder(
        aug_view,
        torch.nan_to_num(aug_view.x.to(device), nan=0.0, posinf=0.0, neginf=0.0),
        motif_budget=0,
        motifs=[],
        communities=[],
        motif_seed=epoch,
    )
    membership_cfg = dict(neg_cfg.get("membership", {}))
    min_nodes_for_node_contrastive = int(membership_cfg.get("min_nodes_for_node_contrastive", 16))
    min_edges_for_node_contrastive = int(membership_cfg.get("min_edges_for_node_contrastive", 2))
    if (
        "contrastive" in disabled
        or hg.num_nodes < min_nodes_for_node_contrastive
        or len(hg.hyperedges) < min_edges_for_node_contrastive
    ):
        losses["contrastive"] = _zero(graph_emb)
    else:
        proj_1 = heads.node_projector(node_emb)
        proj_2 = heads.node_projector(contrastive_node_emb)
        losses["contrastive"] = torch.nan_to_num(
            _cross_view_contrastive_loss(proj_1, proj_2, tau=float(training_cfg.get("contrastive_temperature", 0.07))),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    if "size_pred" in disabled or edge_emb.numel() == 0:
        losses["size_pred"] = _zero(graph_emb)
    else:
        edge_cardinality = incidence.sum(dim=0).float().clamp_min(1.0)
        pred = heads.edge_size_regressor(edge_emb).squeeze(-1)
        target = edge_cardinality.log1p()
        losses["size_pred"] = torch.nan_to_num(F.mse_loss(pred, target), nan=0.0, posinf=0.0, neginf=0.0)

    if "domain_align" not in disabled and float(weight_map.get("domain_align", 0.0)) > 0.0:
        domain_labels = torch.full((node_emb.size(0),), int(hg.metadata.get("domain_id", 0)), device=device, dtype=torch.long)
        losses["domain_align"] = torch.nan_to_num(
            F.cross_entropy(heads.domain_classifier(node_emb), domain_labels),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    else:
        losses["domain_align"] = _zero(graph_emb)
    
    # New: Motif classification task
    num_motif_types = int(training_cfg.get("num_motif_types", 8))
    if "motif" in disabled:
        losses["motif"] = _zero(graph_emb)
    else:
        motif_labels = None
        if hasattr(hg, 'motif_labels') and hg.motif_labels is not None:
            motif_labels = hg.motif_labels.to(device)
        losses["motif"] = torch.nan_to_num(
            compute_motif_classification_loss(node_emb, edge_emb, motif_emb, motif_labels, heads, num_motif_types),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    
    # New: Community prototype alignment task
    num_prototypes = int(training_cfg.get("num_prototypes", 8))
    if "community" in disabled:
        losses["community"] = _zero(graph_emb)
    else:
        losses["community"] = torch.nan_to_num(
            compute_community_alignment_loss(node_emb, community_emb, heads, num_prototypes),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    
    # New: Structure alignment task (cross-view consistency)
    if "structure_align" in disabled:
        losses["structure_align"] = _zero(graph_emb)
    else:
        masked_motif_emb = masked_aux.get("motif_emb", torch.tensor([])).to(device) if masked_aux.get("motif_emb") is not None else motif_emb
        contrastive_motif_emb = contrastive_aux.get("motif_emb", torch.tensor([])).to(device) if contrastive_aux.get("motif_emb") is not None else motif_emb
        losses["structure_align"] = torch.nan_to_num(
            compute_structure_alignment_loss(
                node_emb, edge_emb,
                contrastive_node_emb, contrastive_edge_emb,
                motif_emb, contrastive_motif_emb,
                incidence, heads,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    # === NEW: Disentanglement Losses (Challenge 1) ===
    use_disentanglement = bool(training_cfg.get("use_disentanglement", False))
    if use_disentanglement and hasattr(heads, 'compute_disentanglement_losses'):
        domain_labels = torch.full((node_emb.size(0),), int(hg.metadata.get("domain_id", 0)), device=device, dtype=torch.long)
        
        # Apply disentanglers to get shared/private representations
        z_node_shared, z_node_private, _ = heads.node_disentangler(node_emb)
        z_edge_shared, z_edge_private, _ = heads.edge_disentangler(edge_emb)
        
        # Compute orthogonality and private-domain losses
        dis_losses = heads.compute_disentanglement_losses(
            z_node_shared, z_node_private,
            z_edge_shared, z_edge_private,
            domain_labels,
        )
        
        losses["orth_node"] = dis_losses.get("orth_node", _zero(graph_emb))
        losses["orth_edge"] = dis_losses.get("orth_edge", _zero(graph_emb))
        losses["private_domain_node"] = dis_losses.get("private_domain_node", _zero(graph_emb))
        losses["private_domain_edge"] = dis_losses.get("private_domain_edge", _zero(graph_emb))
        
        # === NEW: Multi-granularity Alignment Losses ===
        use_domain_alignment = bool(training_cfg.get("use_domain_alignment", False))
        if use_domain_alignment and hasattr(heads, 'compute_alignment_losses'):
            # Get confidence-based routing masks
            routing = heads.get_confidence_routing(node_emb, edge_emb, incidence)
            node_align_mask = routing["node"].get("align_mask")
            edge_align_mask = routing["edge"].get("align_mask")
            
            # Compute alignment losses
            align_losses = heads.compute_alignment_losses(
                node_emb=node_emb,
                edge_emb=edge_emb,
                incidence=incidence,
                node_align_mask=node_align_mask,
                edge_align_mask=edge_align_mask,
            )
            
            for key, val in align_losses.items():
                if isinstance(val, torch.Tensor):
                    losses[key] = val
    else:
        losses["orth_node"] = _zero(graph_emb)
        losses["orth_edge"] = _zero(graph_emb)
        losses["private_domain_node"] = _zero(graph_emb)
        losses["private_domain_edge"] = _zero(graph_emb)

    total = _zero(graph_emb)
    for task_name in (
        "masked_node",
        "hyperedge_recon",
        "contrastive",
        "size_pred",
        "domain_align",
        "membership_contrast",
        "motif",
        "community",
        "structure_align",
        "orth_node",
        "orth_edge",
        "private_domain_node",
        "private_domain_edge",
    ):
        total = total + losses[task_name] * float(weight_map.get(task_name, 0.0))
    losses["total"] = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
    return {**losses, "stats": stats}
