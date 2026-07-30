from __future__ import annotations

from typing import Dict, List, Set

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

    masked_view = augment_hypergraph(
        hg,
        feature_mask_rate=float(training_cfg.get("feature_mask_rate", 0.15)),
        edge_dropout_rate=float(training_cfg.get("edge_dropout_rate", 0.2)),
        seed=epoch * 17 + 1,
        strategy="feature_masking",
    )
    masked_node_emb, _, _, _ = encoder(
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
    contrastive_node_emb, _, _, _ = encoder(
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

    incidence = aux["incidence"].to(device)
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

    total = _zero(graph_emb)
    for task_name in (
        "masked_node",
        "hyperedge_recon",
        "contrastive",
        "size_pred",
        "domain_align",
        "membership_contrast",
    ):
        total = total + losses[task_name] * float(weight_map.get(task_name, 1.0))
    losses["total"] = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
    return {**losses, "stats": stats}
