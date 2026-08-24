from __future__ import annotations

import random
from typing import Dict, List, Sequence

import torch
from torch import nn

from utils.hypergraph import SimpleHypergraph


def sample_negative_hyperedges(hg: SimpleHypergraph, num_samples: int, seed: int) -> List[List[int]]:
    generator = torch.Generator().manual_seed(seed)
    negatives: List[List[int]] = []
    if not hg.hyperedges:
        return negatives
    for sample_index in range(num_samples):
        source_edge = hg.hyperedges[sample_index % len(hg.hyperedges)]
        edge_size = len(source_edge)
        if edge_size == 0:
            negatives.append([])
            continue
        candidate = torch.randperm(hg.num_nodes, generator=generator)[:edge_size].tolist()
        negatives.append(sorted(candidate))
    return negatives


def sample_motifs(hg: SimpleHypergraph, budget: int, seed: int) -> List[Dict[str, List[int]]]:
    if not hg.hyperedges or budget <= 0:
        return []
    generator = torch.Generator().manual_seed(seed)
    chosen = torch.randperm(len(hg.hyperedges), generator=generator)[: min(budget, len(hg.hyperedges))].tolist()
    motifs: List[Dict[str, List[int]]] = []
    for seed_edge_index in chosen:
        seed_nodes = set(hg.hyperedges[seed_edge_index])
        edge_ids = [seed_edge_index]
        for candidate_edge_index, edge in enumerate(hg.hyperedges):
            if candidate_edge_index != seed_edge_index and seed_nodes.intersection(edge):
                edge_ids.append(candidate_edge_index)
        motif_nodes = sorted({node for edge_id in edge_ids for node in hg.hyperedges[edge_id]})
        motifs.append({"nodes": motif_nodes, "edges": edge_ids})
    return motifs


def sample_communities(hg: SimpleHypergraph) -> List[Dict[str, List[int]]]:
    incidence = hg.incidence_matrix()
    if incidence.numel() == 0 or hg.num_nodes == 0:
        return []
    N = hg.num_nodes
    device = incidence.device
    sparse_inc = incidence.to_sparse_coo().coalesce() if not incidence.is_sparse else incidence.coalesce()
    node_idx, edge_idx = sparse_inc.indices()

    # Strong co-membership adjacency (overlap >= avg_degree). For small graphs a
    # dense N x N overlap is fine; for large graphs use a *sparse* overlap so we
    # never materialise an N x N dense matrix (which OOMs, e.g. on DBLP).
    if N <= 12000:
        inc_dense = incidence.to_dense()
        overlap = (inc_dense @ inc_dense.t()).float()
        avg_degree = float(inc_dense.sum(dim=1).mean().item())
        threshold = max(1.0, avg_degree)
        adj = overlap >= threshold  # [N, N] bool dense
        a = b = None
    else:
        # Sparse overlap for large graphs. Force float32 + disable autocast:
        # autocast(bf16) would cast the sparse inputs to BFloat16, which CUDA
        # sparse kernels reject, and the incidence may already be BFloat16.
        with torch.autocast(device_type=device.type, enabled=False):
            ov_fp32 = torch.sparse_coo_tensor(
                sparse_inc.indices(), sparse_inc.values().float(), sparse_inc.size())
            overlap_s = torch.sparse.mm(ov_fp32, ov_fp32.transpose(0, 1)).coalesce()
        avg_degree = float(sparse_inc.sum(dim=1).values().mean().item())
        threshold = max(1.0, avg_degree)
        ov_idx = overlap_s.indices()
        ov_mask = overlap_s.values().float() >= threshold
        a = ov_idx[0][ov_mask]
        b = ov_idx[1][ov_mask]
        adj = None

    communities: List[Dict[str, List[int]]] = []
    visited = torch.zeros(N, dtype=torch.bool, device=device)
    remaining = ~visited
    # Outer loop runs once per *community* (not per node); all inner work is
    # vectorised (no O(E) Python set intersection, no N x N dense scan).
    while remaining.any():
        seed = int(torch.nonzero(remaining, as_tuple=False)[0].item())
        if adj is not None:
            members = torch.nonzero(adj[seed]).view(-1)
        else:
            row_mask = a == seed
            members = torch.unique(torch.cat([b[row_mask], a[row_mask], torch.tensor([seed], device=device)]))
        if members.numel() == 0:
            members = torch.tensor([seed], device=device)
        # Edges intersecting the community: edges that have any member node.
        member_mask = torch.isin(node_idx, members)
        edge_ids = torch.unique(edge_idx[member_mask]).tolist()
        communities.append({"nodes": members.tolist(), "edges": edge_ids})
        visited[members] = True
        remaining = ~visited
    return communities


def motif_signatures(hg: SimpleHypergraph, motifs: Sequence[Dict[str, List[int]]]) -> torch.Tensor:
    if not motifs:
        return hg.x.new_zeros((0, 4))
    signatures = []
    for motif in motifs:
        nodes = motif["nodes"]
        edges = motif["edges"]
        edge_sizes = [len(hg.hyperedges[edge_id]) for edge_id in edges] or [1]
        overlap_pairs = 0
        comparisons = 0
        for first_index in range(len(edges)):
            for second_index in range(first_index + 1, len(edges)):
                first_nodes = set(hg.hyperedges[edges[first_index]])
                second_nodes = set(hg.hyperedges[edges[second_index]])
                overlap_pairs += int(bool(first_nodes.intersection(second_nodes)))
                comparisons += 1
        signatures.append(
            [
                float(len(nodes)),
                float(len(edges)),
                float(sum(edge_sizes) / len(edge_sizes)),
                float(overlap_pairs / max(comparisons, 1)),
            ]
        )
    return hg.x.new_tensor(signatures)


def community_signatures(hg: SimpleHypergraph, communities: Sequence[Dict[str, List[int]]]) -> torch.Tensor:
    if not communities:
        return hg.x.new_zeros((0, 4))
    signatures = []
    for community in communities:
        nodes = community["nodes"]
        edges = community["edges"]
        edge_sizes = [len(hg.hyperedges[edge_id]) for edge_id in edges] or [1]
        signatures.append(
            [
                float(len(nodes)),
                float(len(edges)),
                float(sum(edge_sizes) / len(edge_sizes)),
                float(len(edges) / max(len(nodes), 1)),
            ]
        )
    return hg.x.new_tensor(signatures)


def _pool_substructure(
    node_emb: torch.Tensor,
    edge_emb: torch.Tensor,
    nodes: List[int],
    edges: List[int],
    projection: nn.Module,
) -> torch.Tensor:
    if not nodes and not edges:
        return node_emb.new_zeros((projection.out_features,))
    node_pool = node_emb[nodes].mean(dim=0) if nodes else node_emb.new_zeros(node_emb.size(-1))
    edge_pool = edge_emb[edges].mean(dim=0) if edges else node_emb.new_zeros(node_emb.size(-1))
    return projection(torch.cat([node_pool, edge_pool], dim=0))


def build_motif_embeddings(
    node_emb: torch.Tensor,
    edge_emb: torch.Tensor,
    motifs: Sequence[Dict[str, List[int]]],
    projection: nn.Module,
) -> torch.Tensor:
    if not motifs:
        return node_emb.new_zeros((0, projection.out_features))
    return torch.stack(
        [_pool_substructure(node_emb, edge_emb, motif["nodes"], motif["edges"], projection) for motif in motifs],
        dim=0,
    )


def build_community_embeddings(
    node_emb: torch.Tensor,
    edge_emb: torch.Tensor,
    communities: Sequence[Dict[str, List[int]]],
    projection: nn.Module,
) -> torch.Tensor:
    if not communities:
        return node_emb.new_zeros((0, projection.out_features))
    return torch.stack(
        [_pool_substructure(node_emb, edge_emb, community["nodes"], community["edges"], projection) for community in communities],
        dim=0,
    )


def build_cross_scale_embeddings(motif_emb: torch.Tensor, community_emb: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
    pieces = []
    if motif_emb.numel():
        pieces.append(motif_emb)
    if community_emb.numel():
        pieces.append(community_emb)
    pieces.append(graph_emb.unsqueeze(0))
    return torch.cat(pieces, dim=0)


def augment_hypergraph(
    hg: SimpleHypergraph,
    feature_mask_rate: float = 0.15,
    edge_dropout_rate: float = 0.2,
    seed: int = 0,
    strategy: str = "random",
) -> SimpleHypergraph:
    generator = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    kept_edges = [list(edge) for edge in hg.hyperedges]
    masked_x = hg.x.clone()
    metadata = dict(hg.metadata)
    node_mask = torch.zeros(hg.num_nodes, dtype=torch.bool)
    if strategy in {"random", "node_dropping"}:
        drop_count = int(0.15 * hg.num_nodes)
        if drop_count > 0:
            drop_idx = torch.randperm(hg.num_nodes, generator=generator)[:drop_count]
            if strategy == "node_dropping":
                kept_edges = [
                    [node for node in edge if node not in set(int(idx) for idx in drop_idx.tolist())]
                    for edge in kept_edges
                ]
            else:
                masked_x[drop_idx] = 0.0
            node_mask[drop_idx] = True
    if strategy in {"random", "edge_masking"}:
        edge_keep_mask = torch.rand(len(kept_edges), generator=generator) > edge_dropout_rate
        kept_edges = [edge for keep, edge in zip(edge_keep_mask.tolist(), kept_edges) if keep]
    if strategy in {"random", "feature_masking"}:
        feature_mask = torch.rand(masked_x.shape, generator=generator) < feature_mask_rate
        masked_x[feature_mask] = 0.0
        metadata["feature_mask"] = feature_mask
    if strategy == "hyperedge_perturb":
        perturbed = []
        for edge in kept_edges:
            edge_set = set(edge)
            flip_count = max(1, int(0.05 * max(len(edge_set), 1)))
            for _ in range(flip_count):
                candidate = rng.randrange(hg.num_nodes)
                if candidate in edge_set:
                    edge_set.remove(candidate)
                else:
                    edge_set.add(candidate)
            perturbed.append(sorted(edge_set))
        kept_edges = perturbed
    return SimpleHypergraph(
        num_nodes=hg.num_nodes,
        hyperedges=[list(edge) for edge in kept_edges],
        x=masked_x,
        name=f"{hg.name}_aug",
        domain=hg.domain,
        dataset_name=hg.dataset_name,
        node_labels=hg.node_labels.clone(),
        edge_labels=hg.edge_labels[: len(kept_edges)].clone() if hg.edge_labels is not None else None,
        graph_label=hg.graph_label.clone() if hg.graph_label is not None else None,
        node_train_mask=hg.node_train_mask.clone() if hg.node_train_mask is not None else None,
        node_val_mask=hg.node_val_mask.clone() if hg.node_val_mask is not None else None,
        node_test_mask=hg.node_test_mask.clone() if hg.node_test_mask is not None else None,
        metadata={**metadata, "masked_nodes": node_mask},
    )
