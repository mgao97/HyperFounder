from __future__ import annotations

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
    overlap = incidence @ incidence.transpose(0, 1)
    if overlap.numel() == 0:
        return []
    communities: List[Dict[str, List[int]]] = []
    visited = set()
    avg_degree = float(incidence.sum(dim=1).mean().item()) if hg.num_nodes else 0.0
    threshold = max(1.0, avg_degree)
    for node_index in range(hg.num_nodes):
        if node_index in visited:
            continue
        members = torch.where(overlap[node_index] >= threshold)[0].tolist()
        if not members:
            members = [node_index]
        visited.update(members)
        edge_ids = []
        member_set = set(members)
        for edge_index, edge in enumerate(hg.hyperedges):
            if member_set.intersection(edge):
                edge_ids.append(edge_index)
        communities.append({"nodes": sorted(member_set), "edges": edge_ids})
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


def augment_hypergraph(hg: SimpleHypergraph, feature_mask_rate: float, edge_dropout_rate: float, seed: int) -> SimpleHypergraph:
    generator = torch.Generator().manual_seed(seed)
    edge_keep_mask = torch.rand(len(hg.hyperedges), generator=generator) > edge_dropout_rate
    kept_edges = [edge for keep, edge in zip(edge_keep_mask.tolist(), hg.hyperedges) if keep]
    if not kept_edges:
        kept_edges = hg.hyperedges[:1]
    masked_x = hg.x.clone()
    feature_mask = torch.rand(masked_x.shape, generator=generator) < feature_mask_rate
    masked_x[feature_mask] = 0.0
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
        metadata=dict(hg.metadata),
    )
