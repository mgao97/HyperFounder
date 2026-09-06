from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from utils.hypergraph import SimpleHypergraph


@dataclass
class HyperedgeNegativeBatch:
    pos_edge_indices: torch.Tensor
    neg_edge_node_lists: List[List[int]]
    neg_edge_labels: torch.Tensor
    meta: Dict[str, float]


@dataclass
class MembershipNegativeBatch:
    pos_pairs: torch.Tensor
    neg_pairs: torch.Tensor
    hop_labels: Optional[torch.Tensor]
    meta: Dict[str, float]


@dataclass
class SubgraphNegativeBatch:
    pos_subgraph_ids: torch.Tensor
    neg_subgraph_ids: torch.Tensor
    weak_flags: torch.Tensor
    meta: Dict[str, float]


def _build_generator(rng: torch.Generator | int | None = None) -> torch.Generator:
    if isinstance(rng, torch.Generator):
        return rng
    generator = torch.Generator()
    generator.manual_seed(int(rng or 0))
    return generator


def canonicalize_edge(nodes: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted({int(node_id) for node_id in nodes}))


def is_existing_hyperedge(hg: SimpleHypergraph, nodes: Sequence[int]) -> bool:
    candidate = canonicalize_edge(nodes)
    return candidate in {canonicalize_edge(edge) for edge in hg.hyperedges}


def edge_overlap(edge_a: Sequence[int], edge_b: Sequence[int]) -> int:
    return len(set(int(node_id) for node_id in edge_a).intersection(int(node_id) for node_id in edge_b))


def compute_node_to_edge_membership_sets(hg: SimpleHypergraph, max_hop: int) -> Dict[int, Dict[int, List[int]]]:
    # Vectorized BFS over edge-graph using sparse matmul. Falls back to original
    # implementation only if the hypergraph is degenerate.
    incidence = hg.incidence_matrix()
    num_nodes, num_edges = incidence.size(0), incidence.size(1)
    if num_edges == 0 or num_nodes == 0:
        return {node_id: {} for node_id in range(num_nodes)}

    # edge-edge adjacency via incidence^T @ incidence (boolean).
    inc = (incidence > 0).to(torch.float32)
    edge_adj = (inc.transpose(0, 1) @ inc).clamp_max(1.0)
    # Zero the diagonal (no self-loop).
    edge_adj = edge_adj - torch.diag(torch.diagonal(edge_adj))

    # node->incident edges (boolean vector per node, from incidence rows).
    # visited_nodes_to_edges[hop, e] = 1 if edge e is reachable in exactly `hop`
    # node-incident edges.
    visited_per_hop = []
    visited = torch.zeros((num_edges,), dtype=torch.bool)
    next_frontier = (incidence.to(torch.bool)).any(dim=0)  # edges with any node
    # Hop 1 = edges incident to each node? We store per-node reachable sets,
    # so do per-row matmul against edge_adj instead.

    # Per-node BFS over edges using sparse edge_adj.
    memberships: Dict[int, Dict[int, List[int]]] = {node_id: {} for node_id in range(num_nodes)}
    incident_per_node = [(incidence[node_id] > 0).nonzero(as_tuple=True)[0] for node_id in range(num_nodes)]

    for node_id in range(num_nodes):
        incident = incident_per_node[node_id]
        if incident.numel() == 0:
            continue
        memberships[node_id][1] = incident.tolist()
        if max_hop <= 1:
            continue
        visited_e = torch.zeros((num_edges,), dtype=torch.bool)
        visited_e[incident] = True
        frontier = visited_e.clone().to(torch.float32)
        for hop in range(2, max_hop + 1):
            new_frontier = edge_adj @ frontier
            new_frontier = (new_frontier > 0).to(torch.float32)
            new_frontier = new_frontier * (~visited_e).to(torch.float32)
            new_indices = new_frontier.nonzero(as_tuple=True)[0]
            if new_indices.numel() == 0:
                break
            memberships[node_id][hop] = new_indices.tolist()
            visited_e = visited_e | new_frontier.bool()
            frontier = new_frontier
    return memberships


def score_subgraph_structural_strength(subhg: SimpleHypergraph) -> float:
    if subhg.num_nodes <= 0 or not subhg.hyperedges:
        return 0.0
    incidence = subhg.incidence_matrix()
    incidence_density = float(incidence.sum().item()) / max(subhg.num_nodes * len(subhg.hyperedges), 1)
    overlap = incidence.transpose(0, 1) @ incidence
    if overlap.numel() == 0:
        overlap_density = 0.0
    else:
        overlap.fill_diagonal_(0.0)
        overlap_density = float((overlap > 0).float().mean().item())
    component_ratio = float(min(subhg.num_nodes, len(subhg.hyperedges))) / max(subhg.num_nodes, len(subhg.hyperedges), 1)
    motif_presence = float(sum(1 for edge in subhg.hyperedges if len(edge) >= 3)) / max(len(subhg.hyperedges), 1)
    return 0.35 * overlap_density + 0.30 * incidence_density + 0.20 * component_ratio + 0.15 * motif_presence


def is_valid_negative_edge(hg: SimpleHypergraph, candidate_nodes: Sequence[int], cfg: Dict) -> bool:
    candidate = canonicalize_edge(candidate_nodes)
    if len(candidate) < int(cfg.get("min_negative_edge_size", 2)):
        return False
    if len(candidate) != len(list(candidate_nodes)):
        return False
    if any(node_id < 0 or node_id >= hg.num_nodes for node_id in candidate):
        return False
    if is_existing_hyperedge(hg, candidate):
        return False
    return True


def is_valid_negative_subgraph(subhg: SimpleHypergraph, cfg: Dict) -> bool:
    if subhg.num_nodes <= 0 or not subhg.hyperedges:
        return False
    incidence = subhg.incidence_matrix()
    overlap = incidence.transpose(0, 1) @ incidence if incidence.numel() else incidence.new_zeros((0, 0))
    overlap.fill_diagonal_(0.0) if overlap.numel() else None
    overlap_density = float((overlap > 0).float().mean().item()) if overlap.numel() else 0.0
    incidence_density = float(incidence.sum().item()) / max(subhg.num_nodes * len(subhg.hyperedges), 1)
    component_ratio = float(min(subhg.num_nodes, len(subhg.hyperedges))) / max(subhg.num_nodes, len(subhg.hyperedges), 1)
    return (
        overlap_density >= float(cfg.get("min_overlap_density", 0.0))
        and incidence_density >= float(cfg.get("min_incidence_density", 0.0))
        and component_ratio >= float(cfg.get("min_component_ratio", 0.0))
    )


def _sample_replacement_negative(
    hg: SimpleHypergraph,
    anchor_edge: Sequence[int],
    generator: torch.Generator,
    cfg: Dict,
) -> Optional[List[int]]:
    anchor = canonicalize_edge(anchor_edge)
    if len(anchor) < 1:
        return None
    chosen_index = int(torch.randint(0, len(anchor), (1,), generator=generator).item())
    kept = list(anchor)
    replaced_node = kept.pop(chosen_index)
    candidate_pool = [node_id for node_id in range(hg.num_nodes) if node_id not in anchor]
    if not candidate_pool:
        return None
    replacement_index = int(torch.randint(0, len(candidate_pool), (1,), generator=generator).item())
    candidate = kept + [candidate_pool[replacement_index]]
    if replaced_node == candidate[-1]:
        return None
    return sorted(candidate) if is_valid_negative_edge(hg, candidate, cfg) else None


def _sample_overlap_negative(
    hg: SimpleHypergraph,
    edge_index: int,
    generator: torch.Generator,
    cfg: Dict,
    _cache: Dict | None = None,
) -> Optional[List[int]]:
    # Vectorized overlap search via incidence^T @ anchor incidence column.
    # _cache (optional) holds precomputed incidence and overlap_rows to avoid
    # recomputation across multiple calls with the same hypergraph.
    if _cache is None:
        incidence = hg.incidence_matrix()
        anchor_inc = incidence[:, edge_index]
        # overlaps[j] = number of shared nodes between edge_index and edge j.
        overlaps = (incidence.transpose(0, 1) @ anchor_inc.unsqueeze(-1)).squeeze(-1)
        overlaps = overlaps.tolist()
    else:
        overlaps = _cache.get("overlap_rows")[edge_index]
    anchor = canonicalize_edge(hg.hyperedges[edge_index])
    overlapping_indices = [j for j, ov in enumerate(overlaps) if j != edge_index and ov > 0]
    if not overlapping_indices:
        return None
    picked_index = overlapping_indices[int(torch.randint(0, len(overlapping_indices), (1,), generator=generator).item())]
    other_edge = canonicalize_edge(hg.hyperedges[picked_index])
    shared = sorted(set(anchor).intersection(other_edge))
    foreign = [node_id for node_id in other_edge if node_id not in shared]
    candidate = shared[:]
    target_size = len(anchor)
    candidate.extend(foreign[: max(0, target_size - len(candidate))])
    if len(candidate) < target_size:
        node_set = set(candidate)
        fallback = [n for n in range(hg.num_nodes) if n not in node_set]
        if not fallback:
            return None
        n_to_pick = target_size - len(candidate)
        if n_to_pick >= len(fallback):
            candidate = sorted(set(candidate).union(fallback))
        else:
            picks = torch.randint(0, len(fallback), (n_to_pick,), generator=generator).tolist()
            seen = set(candidate)
            for p in picks:
                seen.add(int(fallback[p]))
            candidate = sorted(seen)
        candidate = candidate[:target_size]
    if edge_overlap(anchor, candidate) <= 0:
        return None
    return sorted(candidate) if is_valid_negative_edge(hg, candidate, cfg) else None


def _sample_random_negative(hg: SimpleHypergraph, edge_size: int, generator: torch.Generator, cfg: Dict) -> Optional[List[int]]:
    if edge_size <= 0 or hg.num_nodes <= 0:
        return None
    candidate = torch.randperm(hg.num_nodes, generator=generator)[: min(edge_size, hg.num_nodes)].tolist()
    return sorted(candidate) if is_valid_negative_edge(hg, candidate, cfg) else None


def sample_hyperedge_negatives(
    hg: SimpleHypergraph,
    cfg: Dict,
    rng: torch.Generator | int | None = None,
) -> HyperedgeNegativeBatch:
    generator = _build_generator(rng)
    num_neg_per_pos = int(cfg.get("num_neg_per_pos", 4))
    max_attempts = int(cfg.get("max_attempts", 10))
    modes = [str(mode) for mode in cfg.get("modes", ["replace", "random"])]
    pos_edge_indices: List[int] = []
    neg_edge_node_lists: List[List[int]] = []
    reject_count = 0
    overlap_scores: List[float] = []
    hard_negative_count = 0

    # Precompute incidence-based overlap matrix once (num_edges x num_edges).
    # This lets the overlap-mode sampler skip the O(E^2) Python loop.
    needs_overlap = "overlap" in modes and hg.hyperedges
    if needs_overlap:
        incidence = hg.incidence_matrix()
        overlap_mat = (incidence.transpose(0, 1) @ incidence).tolist()
        overlap_cache = {"overlap_rows": overlap_mat}
    else:
        overlap_cache = None

    for edge_index, edge in enumerate(hg.hyperedges):
        if len(edge) < int(cfg.get("min_negative_edge_size", 2)):
            continue
        for sample_index in range(num_neg_per_pos):
            candidate = None
            selected_mode = modes[sample_index % len(modes)] if modes else "random"
            for _ in range(max_attempts):
                if selected_mode == "replace":
                    candidate = _sample_replacement_negative(hg, edge, generator, cfg)
                elif selected_mode == "overlap":
                    candidate = _sample_overlap_negative(hg, edge_index, generator, cfg, _cache=overlap_cache)
                else:
                    candidate = _sample_random_negative(hg, len(edge), generator, cfg)
                if candidate is not None:
                    break
                reject_count += 1
            if candidate is None:
                continue
            pos_edge_indices.append(edge_index)
            neg_edge_node_lists.append(candidate)
            overlap_value = float(edge_overlap(edge, candidate))
            overlap_scores.append(overlap_value)
            if overlap_value > 0:
                hard_negative_count += 1

    return HyperedgeNegativeBatch(
        pos_edge_indices=torch.tensor(pos_edge_indices, dtype=torch.long),
        neg_edge_node_lists=neg_edge_node_lists,
        neg_edge_labels=torch.zeros(len(neg_edge_node_lists), dtype=torch.float32),
        meta={
            "num_hyperedge_negatives": float(len(neg_edge_node_lists)),
            "hyperedge_negative_rejects": float(reject_count),
            "avg_negative_overlap": float(sum(overlap_scores) / max(len(overlap_scores), 1)),
            "hard_negative_rate": float(hard_negative_count / max(len(neg_edge_node_lists), 1)),
        },
    )


def sample_membership_negatives(
    hg: SimpleHypergraph,
    cfg: Dict,
    rng: torch.Generator | int | None = None,
) -> MembershipNegativeBatch:
    generator = _build_generator(rng)
    num_neg_per_pos = int(cfg.get("num_neg_per_pos", 4))
    max_hop = int(cfg.get("max_membership_hop", 3))
    memberships = compute_node_to_edge_membership_sets(hg, max_hop=max_hop)
    incidence = hg.incidence_matrix()
    pos_pairs: List[List[int]] = []
    neg_pairs: List[List[int]] = []
    hop_labels: List[int] = []
    false_negative_rejects = 0

    # Precompute per-node incident edges (once) and fallback global edge list.
    incident_per_node = [
        (incidence[node_id] > 0).nonzero(as_tuple=True)[0].tolist()
        for node_id in range(hg.num_nodes)
    ]
    all_edge_indices = list(range(len(hg.hyperedges)))

    for node_id in range(hg.num_nodes):
        incident_edges = incident_per_node[node_id]
        if not incident_edges:
            continue
        incident_set = set(incident_edges)
        hop_candidates = memberships.get(node_id, {})
        nearby = sorted({edge_id for hop, edges in hop_candidates.items() if hop >= 2 for edge_id in edges})
        if not nearby:
            continue  # no valid negatives possible; saves fallback list materialization
        for pos_edge_index in incident_edges:
            for _ in range(num_neg_per_pos):
                if nearby:
                    selected = nearby[int(torch.randint(0, len(nearby), (1,), generator=generator).item())]
                    hop = next((hop_id for hop_id, edge_ids in hop_candidates.items() if selected in edge_ids), max_hop)
                else:
                    # Use precomputed all-edge list (one .tolist() per call instead of per-node).
                    fallback = [edge_id for edge_id in all_edge_indices if edge_id not in incident_set]
                    if not fallback:
                        false_negative_rejects += 1
                        continue
                    selected = fallback[int(torch.randint(0, len(fallback), (1,), generator=generator).item())]
                    hop = max_hop
                if selected in incident_set:
                    false_negative_rejects += 1
                    continue
                pos_pairs.append([node_id, pos_edge_index])
                neg_pairs.append([node_id, selected])
                hop_labels.append(int(hop))

    return MembershipNegativeBatch(
        pos_pairs=torch.tensor(pos_pairs, dtype=torch.long) if pos_pairs else torch.zeros((0, 2), dtype=torch.long),
        neg_pairs=torch.tensor(neg_pairs, dtype=torch.long) if neg_pairs else torch.zeros((0, 2), dtype=torch.long),
        hop_labels=torch.tensor(hop_labels, dtype=torch.long) if hop_labels else torch.zeros((0,), dtype=torch.long),
        meta={
            "num_membership_negatives": float(len(neg_pairs)),
            "membership_false_negative_rejects": float(false_negative_rejects),
            "avg_membership_hop": float(sum(hop_labels) / max(len(hop_labels), 1)),
        },
    )


def sample_subhypergraph_negatives(
    subgraphs: Sequence[SimpleHypergraph],
    cfg: Dict,
    rng: torch.Generator | int | None = None,
) -> SubgraphNegativeBatch:
    _ = _build_generator(rng)
    positive_ids: List[int] = []
    negative_ids: List[int] = []
    weak_flags: List[int] = []
    pos_scores: List[float] = []
    neg_scores: List[float] = []

    for index, subgraph in enumerate(subgraphs):
        score = score_subgraph_structural_strength(subgraph)
        if is_valid_negative_subgraph(subgraph, cfg):
            positive_ids.append(index)
            pos_scores.append(score)
        else:
            negative_ids.append(index)
            weak_flags.append(1)
            neg_scores.append(score)

    return SubgraphNegativeBatch(
        pos_subgraph_ids=torch.tensor(positive_ids, dtype=torch.long),
        neg_subgraph_ids=torch.tensor(negative_ids, dtype=torch.long),
        weak_flags=torch.tensor(weak_flags, dtype=torch.bool) if weak_flags else torch.zeros((0,), dtype=torch.bool),
        meta={
            "num_subgraph_negatives": float(len(negative_ids)),
            "avg_subgraph_strength_pos": float(sum(pos_scores) / max(len(pos_scores), 1)),
            "avg_subgraph_strength_neg": float(sum(neg_scores) / max(len(neg_scores), 1)),
        },
    )
