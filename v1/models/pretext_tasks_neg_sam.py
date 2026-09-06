from __future__ import annotations

from typing import Dict, List, Set, Optional

import torch
import torch.nn.functional as F

from v1.models.encoder import UnifiedHypergraphEncoder
from v1.models.heads_neg_sam import TaskHeadsNegSam
from v1.models.negative_sampling_neg_sam import (
    HyperedgeNegativeBatch,
    MembershipNegativeBatch,
    sample_hyperedge_negatives,
    sample_membership_negatives,
)
import os as _os
_USE_HEDG_NEGATIVES = _os.environ.get("USE_HEDG_NEGATIVES", "0") == "1"
if _USE_HEDG_NEGATIVES:
    try:
        from v1.models.hedg_integration import (
            build_hedg_sampler,
            hedg_to_hyperedge_batch,
            hedg_to_membership_batch,
        )
        _HEDG_INTEGRATION_AVAILABLE = True
    except ImportError as _e:
        _HEDG_INTEGRATION_AVAILABLE = False
        print(f"[pretext_tasks_neg_sam] HEDG integration unavailable: {_e}")
else:
    _HEDG_INTEGRATION_AVAILABLE = False
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
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, torch.Tensor | Dict[str, float]]:
    del task_cache
    disabled = drop_tasks or set()
    training_cfg = config.get("training", {})
    neg_cfg = config.get("neg_sampling", {})
    use_task_routing = bool(training_cfg.get("use_task_routing", False))
    # Build a context manager that no-ops when AMP is disabled or CUDA is missing.
    if amp_enabled and torch.cuda.is_available():
        amp_ctx = torch.autocast("cuda", dtype=amp_dtype)
    else:
        from contextlib import nullcontext
        amp_ctx = nullcontext()
    
    x = torch.nan_to_num(hg.x.to(device, non_blocking=True), nan=0.0, posinf=0.0, neginf=0.0)
    with amp_ctx:
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
    with amp_ctx:
        masked_node_emb, masked_edge_emb, masked_graph_emb, masked_aux = encoder(
            masked_view,
            torch.nan_to_num(masked_view.x.to(device, non_blocking=True), nan=0.0, posinf=0.0, neginf=0.0),
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

    # Cache negative samples per (subhypergraph, epoch) so the heavy Python
    # sampling only runs ONCE per subhypergraph per epoch instead of on every
    # training step (which uses the same RNG seed within an epoch).
    neg_cache = hg.metadata.setdefault("_neg_cache", {})
    he_cache_key = (epoch, "hyperedge")
    cached_he = neg_cache.get(he_cache_key)
    if cached_he is not None:
        hyperedge_neg_batch = cached_he
    else:
        if _USE_HEDG_NEGATIVES and _HEDG_INTEGRATION_AVAILABLE and bool(neg_cfg.get("hedg_negatives", {}).get("enabled", False)):
            hedg_cfg = dict(neg_cfg.get("hedg_negatives", {}))
            hedg_sampler = build_hedg_sampler(
                hg, hedg_cfg, seed=epoch * 97 + len(hg.hyperedges) + 1
            )
            hedg_result = hedg_sampler.sample_hyperedge_negatives(
                list(range(len(hg.hyperedges))),
                generator=None,  # sampler uses its own seeded RNG
            )
            hyperedge_neg_batch = hedg_to_hyperedge_batch(hedg_result, hg)
        else:
            hyperedge_neg_batch = sample_hyperedge_negatives(
                hg,
                cfg=dict(neg_cfg.get("hyperedge", {})),
                rng=epoch * 97 + len(hg.hyperedges),
            )
        neg_cache[he_cache_key] = hyperedge_neg_batch
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

    mem_cache_key = (epoch, "membership")
    cached_mem = neg_cache.get(mem_cache_key)
    if cached_mem is not None:
        membership_neg_batch = cached_mem
    else:
        if _USE_HEDG_NEGATIVES and _HEDG_INTEGRATION_AVAILABLE and bool(neg_cfg.get("hedg_negatives", {}).get("enabled", False)):
            hedg_cfg = dict(neg_cfg.get("hedg_negatives", {}))
            hedg_sampler = build_hedg_sampler(
                hg, hedg_cfg, seed=epoch * 131 + hg.num_nodes + 1
            )
            # Use incident edges of each node as positive pairs
            pos_pairs_for_hedg = []
            for n in range(hg.num_nodes):
                for e in (hg.incidence_matrix()[n] > 0).nonzero(as_tuple=True)[0].tolist():
                    pos_pairs_for_hedg.append((n, e))
            hedg_mem_result = hedg_sampler.sample_membership_negatives(
                pos_pairs_for_hedg,
                max_hop=int(hedg_cfg.get("max_membership_hop", 2)),
            )
            membership_neg_batch = hedg_to_membership_batch(hedg_mem_result)
        else:
            membership_neg_batch = sample_membership_negatives(
                hg,
                cfg=dict(neg_cfg.get("membership", {})),
                rng=epoch * 131 + hg.num_nodes,
            )
        neg_cache[mem_cache_key] = membership_neg_batch
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
    with amp_ctx:
        contrastive_node_emb, contrastive_edge_emb, contrastive_graph_emb, contrastive_aux = encoder(
            aug_view,
            torch.nan_to_num(aug_view.x.to(device, non_blocking=True), nan=0.0, posinf=0.0, neginf=0.0),
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

    # Structure discrimination: score original (strong) view higher than the
    # augmented/masked (weak) views via BPR-style ranking. Acts as a regularizer
    # that forces encoder quality-aware output magnitudes. Falls back to 0 if
    # graph_emb tensors are empty (e.g., no valid nodes in subgraph).
    if "structure_discrimination" in disabled:
        losses["structure_discrimination"] = _zero(graph_emb)
    else:
        pos_graph = graph_emb if graph_emb.numel() > 0 else None
        neg_masked = masked_graph_emb if masked_graph_emb.numel() > 0 else None
        neg_contrast = contrastive_graph_emb if contrastive_graph_emb.numel() > 0 else None
        if pos_graph is None or (neg_masked is None and neg_contrast is None):
            losses["structure_discrimination"] = _zero(graph_emb if graph_emb.numel() > 0 else node_emb[:0])
        else:
            pos_graph = F.normalize(pos_graph.unsqueeze(0) if pos_graph.dim() == 1 else pos_graph, dim=-1)
            logit_parts = []
            if neg_masked is not None:
                nm = F.normalize(neg_masked.unsqueeze(0) if neg_masked.dim() == 1 else neg_masked, dim=-1)
                logit_parts.append(-F.logsigmoid((pos_graph * nm).sum(dim=-1).clamp(min=-1.0, max=1.0) * 2.0))
            if neg_contrast is not None:
                nc = F.normalize(neg_contrast.unsqueeze(0) if neg_contrast.dim() == 1 else neg_contrast, dim=-1)
                logit_parts.append(-F.logsigmoid((pos_graph * nc).sum(dim=-1).clamp(min=-1.0, max=1.0) * 2.0))
            stacked = torch.cat(logit_parts, dim=0) if len(logit_parts) > 1 else logit_parts[0]
            losses["structure_discrimination"] = torch.nan_to_num(
                stacked.mean(),
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
        
        def _safe(v):
            t = dis_losses.get(v, _zero(graph_emb))
            if isinstance(t, torch.Tensor):
                return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            return t
        losses["orth_node"] = _safe("orth_node")
        losses["orth_edge"] = _safe("orth_edge")
        losses["private_domain_node"] = _safe("private_domain_node")
        losses["private_domain_edge"] = _safe("private_domain_edge")
        
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
                    losses[key] = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        losses["orth_node"] = _zero(graph_emb)
        losses["orth_edge"] = _zero(graph_emb)
        losses["private_domain_node"] = _zero(graph_emb)
        losses["private_domain_edge"] = _zero(graph_emb)

    total = _zero(graph_emb)
    use_uncertainty = bool(training_cfg.get("use_uncertainty_weighting", False))
    has_uncertainty_params = use_uncertainty and hasattr(heads, "loss_log_sigmas")
    uncertainty_stats = {}
    task_order = (
        "masked_node",
        "hyperedge_recon",
        "contrastive",
        "size_pred",
        "domain_align",
        "membership_contrast",
        "motif",
        "community",
        "structure_align",
        "structure_discrimination",
        "orth_node",
        "orth_edge",
        "private_domain_node",
        "private_domain_edge",
    )
    if has_uncertainty_params:
        for task_name in task_order:
            static_w = float(weight_map.get(task_name, 0.0))
            if static_w <= 0.0:
                continue
            if task_name not in losses:
                continue
            loss_t = losses[task_name]
            if isinstance(loss_t, torch.Tensor):
                if loss_t.numel() == 0:
                    continue
                loss_t = torch.nan_to_num(loss_t, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                loss_t = torch.as_tensor(loss_t, device=graph_emb.device, dtype=graph_emb.dtype)
            if task_name not in heads.loss_log_sigmas:
                total = total + loss_t * static_w
                continue
            log_sigma = heads.loss_log_sigmas[task_name].to(loss_t.dtype).to(loss_t.device)
            precision = torch.exp(-2.0 * log_sigma)
            total = total + 0.5 * precision * loss_t + log_sigma
            sigma_val = float(torch.exp(log_sigma.detach()).cpu().item())
            uncertainty_stats[f"sigma_{task_name}"] = sigma_val
            uncertainty_stats[f"w_{task_name}"] = float(0.5 * precision.detach().cpu().item())
    else:
        for task_name in task_order:
            if task_name in losses:
                lt = losses[task_name]
                if isinstance(lt, torch.Tensor):
                    lt = torch.nan_to_num(lt, nan=0.0, posinf=0.0, neginf=0.0)
                total = total + lt * float(weight_map.get(task_name, 0.0))
    losses["total"] = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
    if uncertainty_stats:
        for k, v in uncertainty_stats.items():
            stats[k] = float(v)
    return {**losses, "stats": stats}
