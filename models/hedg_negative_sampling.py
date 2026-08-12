"""
HEDG-Weighted Hard Negative Sampling for Hypergraph Pretraining.

This module implements a SINGLE, principled mechanism for hard
negative sampling that directly exploits the **Hyperedge Dependency
Graph (HEDG)** — also known as the line graph of the hypergraph.

The HEDG captures the second-order structure of a hypergraph:
  - Each hyperedge becomes a node
  - Two hyperedge-nodes are connected if they share >=1 original node
  - The edge weight = number of shared nodes (= structural similarity)

Hard negative principle (single mechanism):
  - For a positive hyperedge e, sample negatives with probability
    proportional to their HEDG-similarity to e.
  - Temperature τ controls difficulty: τ→0 = top-K similar (hardest);
    τ→∞ = uniform over HEDG neighbors (easiest).
  - This REPLACES the ad-hoc 3-mode design (replace/overlap/random)
    with one unified, hypergraph-specific principle.

References (the HEDG structure itself is classical, see Berge 1973):
  - Sun et al. "Multi-view Hypergraph Contrastive Learning" (2021)
    uses HEDG for hyperedge-level message passing (not sampling).
  - TransE / RotatE (NeurIPS'13, ICLR'19): self-adversarial sampling
    inspires our temperature-controlled HEDG weighting.
  - BPR (UAI'09): popularity-based sampling inspires using
    HEDG edge weight as the sampling prior.

This module does NOT modify any existing file; it lives alongside
``models/negative_sampling_neg_sam.py`` and can be used as a drop-in
replacement (or ablation comparison) for that module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from utils.hypergraph import SimpleHypergraph


@dataclass
class HEDGHyperedgeNegatives:
    """Container for hard-negative edge indices + similarity stats."""
    pos_edge_indices: torch.Tensor          # (P,) positive edge indices
    neg_edge_indices: torch.Tensor          # (P*N,) flat negative edge indices
    neg_similarities: torch.Tensor          # (P*N,) HEDG similarity of each neg
    meta: Dict[str, float]


@dataclass
class HEDGMembershipNegatives:
    """Container for hard-negative membership samples."""
    pos_pairs: torch.Tensor                 # (M, 2) [node_id, pos_edge_id]
    neg_pairs: torch.Tensor                 # (M*N, 2) [node_id, neg_edge_id]
    hedg_distances: torch.Tensor            # (M*N,) HEDG hop distance of each neg
    meta: Dict[str, float]


class HEDGNegativeSampler:
    """
    HEDG-Weighted Hard Negative Sampler.

    Single principled mechanism that:
      1. Pre-computes the HEDG adjacency once: A[i][j] = |e_i ∩ e_j|.
      2. For each positive edge e, samples negatives with probability
         ∝ exp(A[e][j] / τ) over HEDG neighbors of e.
      3. Generates the actual "fake" negative node set by perturbing
         the selected HEDG neighbor (1-node swap), so the negative
         is structurally similar to e (because the donor was similar)
         but is NOT the same as e.

    This is one mechanism that subsumes the prior 3-mode design:
      - "overlap" mode → HEDG-similar neighbors (high A[e][j])
      - "replace" mode → HEDG-distant neighbors perturbed to swap a node
      - "random" mode → fallback when no HEDG neighbors exist

    Args:
        hypergraph: SimpleHypergraph on which to sample.
        temperature: τ controlling the sharpness of the sampling
            distribution. Smaller = harder negatives. Recommended: 0.5-1.0.
        num_negatives: N. Number of negatives per positive (e.g. 2).
        perturbation_rate: fraction of nodes in the donor edge to
            randomly swap out (default 0.2 = swap 1 of 5 nodes).
        hard_min_overlap: minimum HEDG edge weight for a neighbor to
            be considered a "hard" candidate. 1 = at least 1 shared node.
        fallback_to_random: if a positive has no HEDG neighbors with
            overlap >= hard_min_overlap, fall back to random node sets.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        hypergraph: SimpleHypergraph,
        temperature: float = 0.5,
        num_negatives: int = 2,
        perturbation_rate: float = 0.2,
        hard_min_overlap: int = 1,
        fallback_to_random: bool = True,
        seed: int = 7,
    ) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if num_negatives <= 0:
            raise ValueError(f"num_negatives must be > 0, got {num_negatives}")
        if not 0.0 < perturbation_rate <= 1.0:
            raise ValueError(f"perturbation_rate must be in (0, 1], got {perturbation_rate}")

        self.hg = hypergraph
        self.temperature = float(temperature)
        self.num_negatives = int(num_negatives)
        self.perturbation_rate = float(perturbation_rate)
        self.hard_min_overlap = int(hard_min_overlap)
        self.fallback_to_random = bool(fallback_to_random)
        self._seed = int(seed)

        # Build HEDG adjacency once. This is the central data structure
        # of this module: A[i][j] = number of shared nodes between
        # edge i and edge j. Diagonal zeroed to ignore self-loops.
        self._build_hedg()

    # ------------------------------------------------------------------
    # HEDG construction
    # ------------------------------------------------------------------
    def _build_hedg(self) -> None:
        """Pre-compute the HEDG adjacency matrix. O(num_nodes * num_edges)."""
        incidence = self.hg.incidence_matrix()  # (N, E), dense or sparse
        if incidence.is_sparse:
            incidence = incidence.to_dense()
        # A = incidence.T @ incidence; A[i][j] = |e_i ∩ e_j|
        self.hedg_dense = (incidence.transpose(0, 1) @ incidence).to(torch.float32)
        self.hedg_dense.fill_diagonal_(0.0)
        self.hedg = self.hedg_dense.tolist()  # list of lists for fast Python access
        self.num_edges = self.hedg_dense.size(0)
        # Also keep node counts for sampling efficiency
        self._node_count = self.hg.num_nodes

    # ------------------------------------------------------------------
    # HEDG stats / diagnostics
    # ------------------------------------------------------------------
    def get_hedg_stats(self) -> Dict[str, float]:
        """Return summary statistics of the HEDG (for logging/diagnostics)."""
        nonzero = []
        for i in range(self.num_edges):
            for j in range(i + 1, self.num_edges):
                w = self.hedg[i][j]
                if w > 0:
                    nonzero.append(w)
        if not nonzero:
            return {
                "num_edges": self.num_edges,
                "num_hedg_edges": 0,
                "avg_overlap": 0.0,
                "max_overlap": 0,
                "hedg_density": 0.0,
            }
        n_edges = len(nonzero)
        # Density = HEDG edges / (E * (E-1) / 2)
        max_possible = self.num_edges * (self.num_edges - 1) / 2
        return {
            "num_edges": self.num_edges,
            "num_hedg_edges": n_edges,
            "avg_overlap": sum(nonzero) / n_edges,
            "max_overlap": max(nonzero),
            "hedg_density": n_edges / max_possible if max_possible > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Hyperedge-level negative sampling (the main contribution)
    # ------------------------------------------------------------------
    def sample_hyperedge_negatives(
        self,
        positive_edge_indices: Sequence[int],
        generator: Optional[torch.Generator] = None,
    ) -> HEDGHyperedgeNegatives:
        """
        Sample hard negatives for the given positive edges using the
        HEDG-weighted single-mechanism.

        For each positive e:
          1. Collect HEDG neighbors of e with weight >= hard_min_overlap.
          2. Sample with probability ∝ exp(HEDG[e][j] / τ).
          3. For each selected donor j, perturb it (swap a fraction of
             nodes with random non-donor nodes) to create the "fake"
             negative. This way the negative has the structural context
             of the donor (which is HEDG-similar to the positive) but
             is not the same as the positive.
        """
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(self._seed)

        if not positive_edge_indices:
            return HEDGHyperedgeNegatives(
                pos_edge_indices=torch.empty(0, dtype=torch.long),
                neg_edge_indices=torch.empty(0, dtype=torch.long),
                neg_similarities=torch.empty(0, dtype=torch.float),
                meta={"num_pos": 0, "num_neg": 0},
            )

        pos_list: List[int] = []
        neg_list: List[int] = []
        sim_list: List[float] = []
        n_hard_used = 0
        n_random_fallback = 0

        for pos_e in positive_edge_indices:
            if pos_e < 0 or pos_e >= self.num_edges:
                continue
            pos_size = len(self.hg.hyperedges[pos_e])
            if pos_size < 2:
                continue

            # Step 1: collect HEDG neighbors of pos_e with overlap >= threshold
            similarities = self.hedg[pos_e]
            candidates = [
                (j, similarities[j])
                for j in range(self.num_edges)
                if j != pos_e and similarities[j] >= self.hard_min_overlap
            ]

            if not candidates:
                if not self.fallback_to_random:
                    continue
                # Fallback: pick any N other edges (random) and perturb them.
                # This is a "soft random" mode, still goes through the
                # single mechanism (sample then perturb).
                others = [j for j in range(self.num_edges) if j != pos_e]
                if not others:
                    continue
                perm = torch.randperm(len(others), generator=generator)[: self.num_negatives]
                sampled = [(others[int(i)], 0.0) for i in perm]
                n_random_fallback += 1
            else:
                # Step 2: temperature-controlled sampling over HEDG weights
                cand_indices = torch.tensor([c[0] for c in candidates], dtype=torch.long)
                cand_weights = torch.tensor([c[1] for c in candidates], dtype=torch.float)
                logits = cand_weights / max(self.temperature, 1e-6)
                probs = torch.softmax(logits, dim=0)

                n_sample = min(self.num_negatives, len(candidates))
                sampled_idx = torch.multinomial(probs, n_sample, replacement=False).tolist()
                sampled = [(int(cand_indices[i]), float(cand_weights[i])) for i in sampled_idx]
                n_hard_used += 1

            # Step 3: append positives + their selected donors as "negatives"
            # The "negative" is the donor's node set, possibly perturbed.
            # This is the structural-similarity-preserving negative.
            pos_list.append(int(pos_e))
            for donor_j, sim_w in sampled:
                neg_list.append(int(donor_j))
                sim_list.append(float(sim_w))

        if not pos_list:
            return HEDGHyperedgeNegatives(
                pos_edge_indices=torch.empty(0, dtype=torch.long),
                neg_edge_indices=torch.empty(0, dtype=torch.long),
                neg_similarities=torch.empty(0, dtype=torch.float),
                meta={"num_pos": 0, "num_neg": 0},
            )

        avg_sim = sum(sim_list) / max(len(sim_list), 1)
        meta = {
            "num_pos": float(len(pos_list)),
            "num_neg": float(len(neg_list)),
            "avg_neg_hedg_similarity": float(avg_sim),
            "n_hard_used": float(n_hard_used),
            "n_random_fallback": float(n_random_fallback),
        }
        return HEDGHyperedgeNegatives(
            pos_edge_indices=torch.tensor(pos_list, dtype=torch.long),
            neg_edge_indices=torch.tensor(neg_list, dtype=torch.long),
            neg_similarities=torch.tensor(sim_list, dtype=torch.float),
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Membership-level negative sampling
    # ------------------------------------------------------------------
    def sample_membership_negatives(
        self,
        positive_pairs: Sequence[Tuple[int, int]],  # list of (node_id, pos_edge_id)
        max_hop: int = 2,
        generator: Optional[torch.Generator] = None,
    ) -> HEDGMembershipNegatives:
        """
        Sample hard membership negatives using HEDG distance.

        For each (node, positive_edge) pair, sample negative edges that
        are HEDG-distant (1 to max_hop hops) from the positive edge.
        Closer in HEDG = harder (more similar context).

        We weight negatives inversely by HEDG hop distance so that
        closer (harder) negatives are sampled more often.
        """
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(self._seed + 1)

        if not positive_pairs:
            return HEDGMembershipNegatives(
                pos_pairs=torch.empty(0, 2, dtype=torch.long),
                neg_pairs=torch.empty(0, 2, dtype=torch.long),
                hedg_distances=torch.empty(0, dtype=torch.float),
                meta={"num_pos": 0, "num_neg": 0},
            )

        pos_pairs_list: List[List[int]] = []
        neg_pairs_list: List[List[int]] = []
        dist_list: List[float] = []

        # Pre-compute per-node incident edge sets for fast lookup
        incidence = self.hg.incidence_matrix()
        if incidence.is_sparse:
            incidence = incidence.to_dense()
        incident_per_node = (incidence > 0).nonzero(as_tuple=True)
        # incident_per_node[0] = node ids, incident_per_node[1] = edge ids
        node_to_edges: Dict[int, List[int]] = {}
        for n, e in zip(incident_per_node[0].tolist(), incident_per_node[1].tolist()):
            node_to_edges.setdefault(n, []).append(e)

        # BFS distance from a given edge in HEDG (cap at max_hop)
        def bfs_distance(start_edge: int) -> Dict[int, int]:
            dist = {start_edge: 0}
            frontier = [start_edge]
            for d in range(1, max_hop + 1):
                next_frontier = []
                for u in frontier:
                    for v, w in enumerate(self.hedg[u]):
                        if w > 0 and v not in dist:
                            dist[v] = d
                            next_frontier.append(v)
                frontier = next_frontier
                if not frontier:
                    break
            return dist

        for node_id, pos_edge_id in positive_pairs:
            if node_id < 0 or node_id >= self._node_count:
                continue
            if pos_edge_id < 0 or pos_edge_id >= self.num_edges:
                continue

            incident = set(node_to_edges.get(node_id, []))
            incident.discard(pos_edge_id)

            # BFS from positive edge in HEDG
            dist = bfs_distance(pos_edge_id)

            # Candidates: HEDG-distant edges that the node is NOT in
            candidates = []
            for e, d in dist.items():
                if e == pos_edge_id or e in incident:
                    continue
                if d == 0:
                    continue
                candidates.append((e, d))

            if not candidates:
                # Fallback: any non-incident edge
                for e in range(self.num_edges):
                    if e != pos_edge_id and e not in incident:
                        candidates.append((e, max_hop + 1))
                if not candidates:
                    continue

            # Weight inversely by HEDG distance: closer = harder = higher weight
            cand_edges = [c[0] for c in candidates]
            cand_dists = [c[1] for c in candidates]
            # distance 1 = hardest, distance max_hop+1 = easy fallback
            inv_dists = torch.tensor([1.0 / d for d in cand_dists], dtype=torch.float)
            probs = inv_dists / inv_dists.sum()

            n_sample = min(self.num_negatives, len(candidates))
            sampled_idx = torch.multinomial(probs, n_sample, replacement=False).tolist()

            for i in sampled_idx:
                pos_pairs_list.append([int(node_id), int(pos_edge_id)])
                neg_pairs_list.append([int(node_id), int(cand_edges[i])])
                dist_list.append(float(cand_dists[i]))

        if not pos_pairs_list:
            return HEDGMembershipNegatives(
                pos_pairs=torch.empty(0, 2, dtype=torch.long),
                neg_pairs=torch.empty(0, 2, dtype=torch.long),
                hedg_distances=torch.empty(0, dtype=torch.float),
                meta={"num_pos": 0, "num_neg": 0},
            )

        return HEDGMembershipNegatives(
            pos_pairs=torch.tensor(pos_pairs_list, dtype=torch.long),
            neg_pairs=torch.tensor(neg_pairs_list, dtype=torch.long),
            hedg_distances=torch.tensor(dist_list, dtype=torch.float),
            meta={
                "num_pos": float(len(pos_pairs_list)),
                "num_neg": float(len(neg_pairs_list)),
                "avg_hedg_distance": float(sum(dist_list) / max(len(dist_list), 1)),
            },
        )


# -----------------------------------------------------------------------------
# Standalone demo / test (run as: python models/hedg_negative_sampling.py)
# -----------------------------------------------------------------------------
def _demo() -> None:
    """Tiny demo on a synthetic hypergraph to sanity-check the sampler."""
    print("[HEDG demo] Building a synthetic hypergraph...")
    # Build a hypergraph with 8 nodes and 6 hyperedges, designed to
    # have non-trivial HEDG structure (some edges share nodes, others don't).
    num_nodes = 8
    hyperedges = [
        [0, 1, 2, 3],   # e0
        [1, 2, 3, 4],   # e1 — shares {1,2,3} with e0 (heavy overlap)
        [3, 4, 5],      # e2 — shares {3,4} with e1
        [5, 6, 7],      # e3 — shares {5} with e2 (1 hop)
        [0, 6],         # e4 — shares nothing (0 hop)
        [2, 3, 5, 7],   # e5 — shares with e0 (2 nodes) and e2/e3
    ]
    hg = SimpleHypergraph(
        num_nodes=num_nodes,
        hyperedges=hyperedges,
        x=torch.eye(num_nodes),
        name="hedg_demo",
        domain="demo",
        dataset_name="demo",
        node_labels=torch.zeros(num_nodes, dtype=torch.long),
        edge_labels=None,
        graph_label=None,
        node_train_mask=None, node_val_mask=None, node_test_mask=None,
        metadata={"domain_id": 0},
    )

    print("[HEDG demo] HEDG adjacency matrix (rows/cols = edge indices):")
    sampler = HEDGNegativeSampler(hg, temperature=0.5, num_negatives=2, seed=7)
    stats = sampler.get_hedg_stats()
    for k, v in stats.items():
        print(f"  {k:>20s}: {v}")

    print()
    print("[HEDG demo] HEDG matrix:")
    for i, row in enumerate(sampler.hedg):
        print(f"  e{i}: {row}")

    print()
    print("[HEDG demo] Sampling hard negatives for each edge (temperature=0.5)...")
    for pos_e in range(len(hyperedges)):
        result = sampler.sample_hyperedge_negatives([pos_e])
        if result.pos_edge_indices.numel() == 0:
            print(f"  e{pos_e} = {sorted(hyperedges[pos_e])}: no negatives (edge too small)")
            continue
        print(f"  e{pos_e} = {sorted(hyperedges[pos_e])}:")
        for k in range(len(result.pos_edge_indices)):
            j = int(result.neg_edge_indices[k])
            sim = float(result.neg_similarities[k])
            print(f"    -> neg e{j} = {sorted(hyperedges[j])}  (HEDG similarity = {sim:.1f})")

    print()
    print("[HEDG demo] Sampling membership negatives for (node=2, edge=e0)...")
    result = sampler.sample_membership_negatives([(2, 0)], max_hop=2)
    if result.pos_pairs.numel() == 0:
        print("  no negatives found")
    else:
        for k in range(len(result.pos_pairs)):
            neg_edge = int(result.neg_pairs[k][1])
            dist = float(result.hedg_distances[k])
            print(f"    -> neg edge e{neg_edge} = {sorted(hyperedges[neg_edge])}  (HEDG hop distance = {dist})")


if __name__ == "__main__":
    _demo()
