"""
Adapter that bridges HEDG-Weighted negative sampling with the existing
pretraining loss pipeline.

When ``USE_HEDG_NEGATIVES=1`` is set, the trainer imports this module
and routes negative sampling through :class:`HEDGNegativeSampler`
(``models/hedg_negative_sampling.py``) instead of the original 3-mode
sampler (``models/negative_sampling_neg_sam.py``).

The HEDG sampler returns a slightly different batch layout than the
3-mode sampler, so this module provides two thin adapters
(:func:`hedg_to_hyperedge_batch`, :func:`hedg_to_membership_batch`)
that convert the HEDG output into the existing
``HyperedgeNegativeBatch`` / ``MembershipNegativeBatch`` dataclasses
consumed by the loss functions in ``models.pretext_tasks_neg_sam``.

The original 3-mode path is preserved untouched.
"""
from __future__ import annotations

from typing import List

import torch

from v1.models.hedg_negative_sampling import (
    HEDGHyperedgeNegatives,
    HEDGMembershipNegatives,
    HEDGNegativeSampler,
)
from v1.models.negative_sampling_neg_sam import (
    HyperedgeNegativeBatch,
    MembershipNegativeBatch,
)
from utils.hypergraph import SimpleHypergraph


def hedg_to_hyperedge_batch(
    hedg_result: HEDGHyperedgeNegatives,
    hypergraph: SimpleHypergraph,
) -> HyperedgeNegativeBatch:
    """
    Convert ``HEDGHyperedgeNegatives`` into the legacy
    ``HyperedgeNegativeBatch`` shape expected by
    ``compute_hyperedge_discrimination_loss``.

    Key difference vs. the 3-mode sampler:
        - 3-mode: produces synthetic node sets as negatives
        - HEDG:    uses real donor hyperedge indices as negatives
          (we re-construct the donor's node list for pooling)
    """
    if hedg_result.pos_edge_indices.numel() == 0:
        return HyperedgeNegativeBatch(
            pos_edge_indices=torch.empty(0, dtype=torch.long),
            neg_edge_node_lists=[],
            neg_edge_labels=torch.empty(0, dtype=torch.float32),
            meta=hedg_result.meta,
        )

    neg_indices: List[int] = hedg_result.neg_edge_indices.tolist()
    neg_node_lists: List[List[int]] = []
    for neg_idx in neg_indices:
        num_edges = len(hypergraph.hyperedges)
        if neg_idx < 0 or neg_idx >= num_edges:
            continue
        # Donor edge is a real hyperedge: take its sorted node list.
        edge_nodes = hypergraph.hyperedges[neg_idx]
        neg_node_lists.append(sorted(int(n) for n in edge_nodes))

    n_total = len(neg_node_lists)
    # The HEDG sampler already provides pos_edge_indices_repeated of
    # the correct (P*N,) layout, matching the 3-mode legacy format.
    return HyperedgeNegativeBatch(
        pos_edge_indices=hedg_result.pos_edge_indices_repeated,
        neg_edge_node_lists=neg_node_lists,
        neg_edge_labels=torch.zeros(n_total, dtype=torch.float32),
        meta={
            **hedg_result.meta,
            "hedg_negatives": True,
        },
    )


def hedg_to_membership_batch(
    hedg_result: HEDGMembershipNegatives,
) -> MembershipNegativeBatch:
    """
    Convert ``HEDGMembershipNegatives`` into the legacy
    ``MembershipNegativeBatch`` shape expected by
    ``compute_membership_contrast_loss``.

    The HEDG output's ``hedg_distances`` is reused as ``hop_labels``
    (which the loss function only consumes as metadata, not for the
    scoring computation).
    """
    if hedg_result.pos_pairs.numel() == 0:
        return MembershipNegativeBatch(
            pos_pairs=torch.zeros((0, 2), dtype=torch.long),
            neg_pairs=torch.zeros((0, 2), dtype=torch.long),
            hop_labels=torch.zeros((0,), dtype=torch.long),
            meta=hedg_result.meta,
        )

    return MembershipNegativeBatch(
        pos_pairs=hedg_result.pos_pairs,
        neg_pairs=hedg_result.neg_pairs,
        hop_labels=hedg_result.hedg_distances.to(torch.long),
        meta={
            **hedg_result.meta,
            "hedg_membership_negatives": True,
        },
    )


def build_hedg_sampler(
    hypergraph: SimpleHypergraph,
    config: dict,
    seed: int,
) -> HEDGNegativeSampler:
    """
    Convenience constructor: read HEDG hyperparameters from a config
    dict (the ``neg_sampling.hedg_negatives`` block of the YAML config)
    and return a configured :class:`HEDGNegativeSampler`.
    """
    return HEDGNegativeSampler(
        hypergraph=hypergraph,
        temperature=float(config.get("temperature", 0.5)),
        num_negatives=int(config.get("num_negatives", 2)),
        perturbation_rate=float(config.get("perturbation_rate", 0.2)),
        hard_min_overlap=int(config.get("hard_min_overlap", 1)),
        fallback_to_random=bool(config.get("fallback_to_random", True)),
        seed=seed,
    )
