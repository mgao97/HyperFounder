from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans

from utils.hypergraph import SimpleHypergraph
from utils.sampling import community_signatures, motif_signatures, sample_communities, sample_motifs


def _safe_kmeans(features: np.ndarray, num_clusters: int, seed: int) -> np.ndarray:
    if len(features) == 0:
        return np.zeros((0,), dtype=np.int64)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    actual_clusters = max(1, min(num_clusters, len(features)))
    model = KMeans(n_clusters=actual_clusters, n_init=10, random_state=seed)
    return model.fit_predict(features)


def node_descriptors(hg: SimpleHypergraph) -> torch.Tensor:
    incidence = hg.incidence_matrix()
    node_degree = incidence.sum(dim=1)
    edge_sizes = incidence.sum(dim=0)
    neighbor_overlap = incidence @ incidence.transpose(0, 1)
    two_hop = (neighbor_overlap > 0).float().sum(dim=1) - 1.0

    descriptors: List[torch.Tensor] = []
    for node_id in range(hg.num_nodes):
        incident_edges = incidence[node_id] > 0
        incident_sizes = edge_sizes[incident_edges]
        if incident_sizes.numel() == 0:
            incident_mean = node_degree.new_tensor(0.0)
            incident_max = node_degree.new_tensor(0.0)
        else:
            incident_mean = incident_sizes.mean()
            incident_max = incident_sizes.max()
        overlap_stat = neighbor_overlap[node_id].sum() - neighbor_overlap[node_id, node_id]
        descriptors.append(
            torch.stack(
                [
                    node_degree[node_id],
                    incident_mean,
                    incident_max,
                    two_hop[node_id],
                    overlap_stat,
                ]
            )
        )
    return torch.stack(descriptors, dim=0)


def build_node_pseudo_labels(hg: SimpleHypergraph, num_clusters: int, seed: int) -> torch.Tensor:
    descriptors = node_descriptors(hg).cpu().numpy()
    labels = _safe_kmeans(descriptors, num_clusters=num_clusters, seed=seed)
    return torch.tensor(labels, dtype=torch.long)


def build_motif_pseudo_labels(
    hg: SimpleHypergraph,
    motif_budget: int,
    num_clusters: int,
    seed: int,
) -> Tuple[List[Dict[str, List[int]]], torch.Tensor, torch.Tensor]:
    motifs = sample_motifs(hg, budget=motif_budget, seed=seed)
    signatures = motif_signatures(hg, motifs)
    labels = _safe_kmeans(signatures.cpu().numpy(), num_clusters=num_clusters, seed=seed)
    return motifs, signatures, torch.tensor(labels, dtype=torch.long)


def build_community_pseudo_labels(
    hg: SimpleHypergraph,
    num_clusters: int,
    seed: int,
) -> Tuple[List[Dict[str, List[int]]], torch.Tensor, torch.Tensor, torch.Tensor]:
    communities = sample_communities(hg)
    signatures = community_signatures(hg, communities)
    community_labels = _safe_kmeans(signatures.cpu().numpy(), num_clusters=num_clusters, seed=seed)
    node_labels = torch.zeros((hg.num_nodes,), dtype=torch.long)
    for community_index, community in enumerate(communities):
        label = int(community_labels[community_index]) if len(community_labels) else 0
        for node_id in community["nodes"]:
            node_labels[node_id] = label
    return communities, signatures, torch.tensor(community_labels, dtype=torch.long), node_labels


def refresh_cross_domain_prototypes(
    motif_embeddings: Sequence[torch.Tensor],
    num_clusters: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not motif_embeddings:
        empty = torch.zeros((0, 1))
        return empty, torch.zeros((0,), dtype=torch.long)

    stacked = torch.cat(motif_embeddings, dim=0)
    labels = _safe_kmeans(stacked.detach().cpu().numpy(), num_clusters=num_clusters, seed=seed)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=stacked.device)

    centers = []
    for cluster_id in range(label_tensor.max().item() + 1):
        mask = label_tensor == cluster_id
        centers.append(stacked[mask].mean(dim=0))
    return torch.stack(centers, dim=0), label_tensor
