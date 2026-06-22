from __future__ import annotations

from typing import Dict, Set

import torch
import torch.nn.functional as F

from models.encoder import UnifiedHypergraphEncoder
from models.heads import TaskHeads
from utils.hypergraph import SimpleHypergraph
from utils.sampling import augment_hypergraph, sample_negative_hyperedges


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_tensor(0.0)


def _global_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(1.0 - F.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0)).mean(), nan=0.0, posinf=0.0, neginf=0.0)


def _structure_loss(
    heads: TaskHeads,
    node_emb: torch.Tensor,
    edge_emb: torch.Tensor,
    incidence: torch.Tensor,
) -> torch.Tensor:
    if incidence.numel() == 0 or edge_emb.numel() == 0:
        return _zero(node_emb.mean(dim=0))
    incident_pairs = torch.nonzero(incidence > 0, as_tuple=False)
    if incident_pairs.numel() == 0:
        return _zero(node_emb.mean(dim=0))
    node_repr = heads.structure_projection(node_emb[incident_pairs[:, 0]])
    edge_repr = edge_emb[incident_pairs[:, 1]]
    return F.mse_loss(node_repr, edge_repr)


def compute_pretraining_losses(
    encoder: UnifiedHypergraphEncoder,
    heads: TaskHeads,
    hg: SimpleHypergraph,
    task_cache: Dict,
    config: Dict,
    device: torch.device,
    epoch: int,
    drop_tasks: Set[str] | None = None,
) -> Dict[str, torch.Tensor]:
    disabled = drop_tasks or set()
    x = torch.nan_to_num(hg.x.to(device), nan=0.0, posinf=0.0, neginf=0.0)
    node_emb, edge_emb, graph_emb, aux = encoder(
        hg,
        x,
        motif_budget=int(config["training"]["motif_budget"]),
        motifs=task_cache["motifs"],
        communities=task_cache["communities"],
        motif_seed=epoch,
    )
    losses: Dict[str, torch.Tensor] = {}
    weight_map = config["training"]["loss_weights"]
    losses["struct"] = _zero(graph_emb) if "struct" in disabled else _structure_loss(heads, node_emb, edge_emb, aux["incidence"])

    if "node" in disabled:
        losses["node"] = _zero(graph_emb)
    else:
        node_logits = heads.node_head(node_emb)
        losses["node"] = torch.nan_to_num(F.cross_entropy(node_logits, task_cache["node_labels"].to(device)), nan=0.0, posinf=0.0, neginf=0.0)

    if "edge" in disabled or edge_emb.numel() == 0:
        losses["edge"] = _zero(graph_emb)
    else:
        negative_edges = sample_negative_hyperedges(
            hg,
            num_samples=len(hg.hyperedges),
            seed=epoch + len(hg.hyperedges),
        )
        negative_emb = encoder.encode_candidate_hyperedges(node_emb, negative_edges)
        positive_logits = heads.edge_head(edge_emb).squeeze(-1)
        negative_logits = heads.edge_head(negative_emb).squeeze(-1)
        logits = torch.cat([positive_logits, negative_logits], dim=0)
        labels = torch.cat(
            [
                torch.ones_like(positive_logits),
                torch.zeros_like(negative_logits),
            ],
            dim=0,
        )
        losses["edge"] = torch.nan_to_num(F.binary_cross_entropy_with_logits(logits, labels), nan=0.0, posinf=0.0, neginf=0.0)

    motif_emb = aux["motif_emb"]
    if "motif" in disabled or motif_emb.numel() == 0:
        losses["motif"] = _zero(graph_emb)
    else:
        motif_logits = heads.motif_head(motif_emb)
        losses["motif"] = torch.nan_to_num(F.cross_entropy(motif_logits, task_cache["motif_labels"].to(device)), nan=0.0, posinf=0.0, neginf=0.0)

    if "community" in disabled:
        losses["community"] = _zero(graph_emb)
    else:
        community_logits = heads.community_head(node_emb)
        losses["community"] = torch.nan_to_num(F.cross_entropy(community_logits, task_cache["community_node_labels"].to(device)), nan=0.0, posinf=0.0, neginf=0.0)

    if "global" in disabled:
        losses["global"] = _zero(graph_emb)
    else:
        aug_1 = augment_hypergraph(
            hg,
            feature_mask_rate=float(config["training"]["feature_mask_rate"]),
            edge_dropout_rate=float(config["training"]["edge_dropout_rate"]),
            seed=epoch * 7 + 1,
        )
        aug_2 = augment_hypergraph(
            hg,
            feature_mask_rate=float(config["training"]["feature_mask_rate"]),
            edge_dropout_rate=float(config["training"]["edge_dropout_rate"]),
            seed=epoch * 7 + 2,
        )
        _, _, graph_emb_1, _ = encoder(aug_1, aug_1.x.to(device), motif_budget=0, motifs=[], motif_seed=epoch)
        _, _, graph_emb_2, _ = encoder(aug_2, aug_2.x.to(device), motif_budget=0, motifs=[], motif_seed=epoch)
        proj_1 = heads.graph_projector(graph_emb_1)
        proj_2 = heads.graph_projector(graph_emb_2)
        losses["global"] = _global_contrastive_loss(proj_1, proj_2)

    cross_emb = aux["cross_emb"]
    if "cross" in disabled or cross_emb.numel() == 0 or task_cache["prototype_labels"].numel() == 0:
        losses["cross"] = _zero(graph_emb)
    else:
        prototype_logits = heads.prototype_head(cross_emb)
        losses["cross"] = torch.nan_to_num(F.cross_entropy(prototype_logits, task_cache["prototype_labels"].to(device)), nan=0.0, posinf=0.0, neginf=0.0)

    total = _zero(graph_emb)
    for task_name in ("struct", "node", "edge", "motif", "community", "global", "cross"):
        total = total + losses[task_name] * float(weight_map.get(task_name, 1.0))
    losses["total"] = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
    return losses
