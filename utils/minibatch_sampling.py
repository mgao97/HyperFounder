from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from utils.hypergraph import SimpleHypergraph


@dataclass
class SubhypergraphMeta:
    """Enhanced metadata for sampled sub-hypergraphs."""
    # Basic counts
    num_nodes: int
    num_edges: int
    
    # Structural statistics
    incidence_nnz: int
    component_ratio: float
    overlap_density: float
    cardinality_mean: float
    cardinality_std: float
    
    # Quality assessment
    validity_flag: bool
    quality_score: float
    
    # Routing decision
    routing: Dict[str, bool]
    
    # Additional info
    domain_id: int
    parent_graph_name: str
    
    def to_dict(self) -> Dict:
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "incidence_nnz": self.incidence_nnz,
            "component_ratio": self.component_ratio,
            "overlap_density": self.overlap_density,
            "cardinality_mean": self.cardinality_mean,
            "cardinality_std": self.cardinality_std,
            "validity_flag": self.validity_flag,
            "quality_score": self.quality_score,
            "routing": self.routing,
            "domain_id": self.domain_id,
        }


def compute_subhypergraph_quality(
    hg: SimpleHypergraph,
    min_nodes: int = 4,
    min_edges: int = 2,
    tau_hyperedge: float = 0.55,
    tau_motif: float = 0.70,
    tau_membership: float = 0.40,
    tau_hard_negative: float = 0.40,
    domain_id: int = 0,
) -> SubhypergraphMeta:
    """
    Compute quality score and metadata for a sampled sub-hypergraph.
    
    Optimized: collapse all .item() calls into a single CPU sync.
    """
    num_nodes = hg.num_nodes
    num_edges = len(hg.hyperedges)
    if num_edges == 0 or num_nodes == 0:
        # Fast path for degenerate graphs - avoid any GPU work.
        return SubhypergraphMeta(
            num_nodes=num_nodes,
            num_edges=num_edges,
            incidence_nnz=0,
            component_ratio=0.0,
            overlap_density=0.0,
            cardinality_mean=0.0,
            cardinality_std=0.0,
            validity_flag=False,
            quality_score=0.0,
            routing=_get_task_routing_decision(
                quality_score=0.0, validity_flag=False, num_nodes=num_nodes,
                tau_hyperedge=tau_hyperedge, tau_motif=tau_motif,
                tau_membership=tau_membership, tau_hard_negative=tau_hard_negative,
            ),
            domain_id=domain_id,
            parent_graph_name=hg.name,
        )
    
    incidence = hg.incidence_matrix()
    dense_inc = incidence.to_dense() if incidence.is_sparse else incidence
    inc_bool = dense_inc > 0  # (N, E) bool
    edge_card = inc_bool.sum(dim=0).float()  # (E,)
    
    # All statistics as tensors until a single sync at the end.
    incidence_nnz_t = inc_bool.sum()
    nodes_with_edges_t = (edge_card > 0).sum()
    
    overlap_density = 0.0
    if num_edges >= 2:
        # Boolean matmul: overlap_matrix[i,j] = number of shared nodes.
        overlap_matrix = dense_inc.transpose(0, 1) @ dense_inc
        # Use the upper-triangular off-diagonal mean to avoid the explicit triu_indices call.
        ones = torch.ones_like(overlap_matrix)
        mask = torch.triu(ones, diagonal=1)
        edge_overlaps_sum = (overlap_matrix * mask).sum()
        pair_count = max(num_edges * (num_edges - 1) // 2, 1)
        overlap_density_t = edge_overlaps_sum / pair_count
        overlap_density_norm_t = overlap_density_t / max(num_nodes, 1)
    else:
        overlap_density_norm_t = torch.tensor(0.0, device=dense_inc.device)
    
    cardinality_mean_t = edge_card.mean() if edge_card.numel() > 0 else torch.tensor(0.0, device=edge_card.device)
    cardinality_std_t = edge_card.std() if edge_card.numel() > 1 else torch.tensor(0.0, device=edge_card.device)
    
    # SINGLE CPU sync to read all scalars.
    stats = torch.stack([
        incidence_nnz_t.float(),
        nodes_with_edges_t.float(),
        overlap_density_norm_t.float() if isinstance(overlap_density_norm_t, torch.Tensor) else torch.tensor(float(overlap_density_norm_t)),
        cardinality_mean_t,
        cardinality_std_t,
    ]).cpu().tolist()
    incidence_nnz, nodes_with_edges, overlap_density, cardinality_mean, cardinality_std = stats
    overlap_density = min(float(overlap_density), 1.0)
    
    component_ratio = nodes_with_edges / num_nodes
    validity_flag = (
        num_nodes >= min_nodes
        and num_edges >= min_edges
        and incidence_nnz > 0
    )
    quality_score = _compute_quality_score(
        validity_flag=bool(validity_flag),
        num_nodes=num_nodes,
        num_edges=num_edges,
        component_ratio=component_ratio,
        overlap_density=overlap_density,
        incidence_nnz=int(incidence_nnz),
    )
    routing = _get_task_routing_decision(
        quality_score=quality_score,
        validity_flag=bool(validity_flag),
        num_nodes=num_nodes,
        tau_hyperedge=tau_hyperedge,
        tau_motif=tau_motif,
        tau_membership=tau_membership,
        tau_hard_negative=tau_hard_negative,
    )
    return SubhypergraphMeta(
        num_nodes=num_nodes,
        num_edges=num_edges,
        incidence_nnz=incidence_nnz,
        component_ratio=component_ratio,
        overlap_density=overlap_density,
        cardinality_mean=cardinality_mean,
        cardinality_std=cardinality_std,
        validity_flag=validity_flag,
        quality_score=quality_score,
        routing=routing,
        domain_id=domain_id,
        parent_graph_name=hg.name,
    )


def _compute_quality_score(
    validity_flag: bool,
    num_nodes: int,
    num_edges: int,
    component_ratio: float,
    overlap_density: float,
    incidence_nnz: int,
) -> float:
    """Compute combined quality score."""
    if not validity_flag:
        return 0.0
    
    # Size score
    size_score = min(num_nodes / 32.0, 1.0) * min(num_edges / 8.0, 1.0)
    size_score = min(size_score, 1.0)
    
    # Connectivity score
    connectivity_score = min(component_ratio / 0.9, 1.0)
    
    # Overlap score
    overlap_score = min(overlap_density / 0.1, 1.0)
    
    # Density score
    max_nnz = num_nodes * num_edges if num_nodes > 0 and num_edges > 0 else 1
    density_score = incidence_nnz / max(max_nnz, 1)
    
    # Weighted combination
    quality = (
        0.25 * size_score +
        0.30 * connectivity_score +
        0.25 * overlap_score +
        0.20 * density_score
    )
    
    return min(max(quality, 0.0), 1.0)


def _get_task_routing_decision(
    quality_score: float,
    validity_flag: bool,
    num_nodes: int,
    tau_hyperedge: float = 0.55,
    tau_motif: float = 0.70,
    tau_membership: float = 0.40,
    tau_hard_negative: float = 0.40,
) -> Dict[str, bool]:
    """Determine which tasks to apply based on quality."""
    routing = {
        "valid": validity_flag,
        "membership": False,
        "hyperedge_recon": False,
        "contrastive": False,
        "motif": False,
        "community": False,
        "structure_discrimination": False,
        "hard_negative": False,
        "exclude": not validity_flag,
    }
    
    if not validity_flag:
        return routing
    
    # Hard negative: weak but valid
    if quality_score < tau_hard_negative:
        routing["hard_negative"] = True
        routing["exclude"] = False
        return routing
    
    # Membership
    if quality_score >= tau_membership and num_nodes >= 2:
        routing["membership"] = True
    
    # Hyperedge recon + contrastive
    if quality_score >= tau_hyperedge:
        routing["hyperedge_recon"] = True
        routing["contrastive"] = True
    
    # Motif/community
    if quality_score >= tau_motif:
        routing["motif"] = True
        routing["community"] = True
        routing["structure_discrimination"] = True
    
    return routing


def sample_seed_hyperedges(hg: SimpleHypergraph, num_seeds: int, seed: int) -> List[int]:
    if not hg.hyperedges or num_seeds <= 0:
        return []
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(hg.hyperedges), generator=generator)
    return permutation[: min(num_seeds, len(hg.hyperedges))].tolist()


def _rank_frontier_edges(
    hg: SimpleHypergraph,
    candidate_edges: Sequence[int],
    selected_nodes: set[int],
    edge_sets: Optional[List[frozenset]] = None,
    seed: int = 0,
) -> List[int]:
    """Rank frontier edges by overlap with selected nodes.
    
    Args:
        hg: Hypergraph
        candidate_edges: List of candidate edge IDs
        selected_nodes: Set of already selected node IDs
        edge_sets: Pre-computed frozenset list for edges (optional, for speed)
        seed: Random seed
    """
    if not candidate_edges:
        return []
    n = len(candidate_edges)
    overlaps = [0] * n
    for i, edge_id in enumerate(candidate_edges):
        if edge_sets is not None:
            overlaps[i] = len(edge_sets[edge_id] & selected_nodes)
        else:
            overlaps[i] = len(selected_nodes.intersection(hg.hyperedges[edge_id]))
    
    generator = torch.Generator().manual_seed(seed)
    noise = torch.rand(n, generator=generator)
    
    # Use argsort for efficiency
    indices = torch.argsort(-torch.tensor(overlaps, dtype=torch.float32), stable=True)
    # Secondary sort by noise (ascending)
    noise_sorted_indices = torch.argsort(noise[indices])
    ranked = [candidate_edges[indices[noise_sorted_indices[j]].item()] for j in range(n)]
    return ranked


def induce_sampled_subhypergraph(
    hg: SimpleHypergraph,
    node_ids: Sequence[int],
    edge_ids: Sequence[int],
    seed_edge_ids: Sequence[int],
    sampling_depth: int,
) -> SimpleHypergraph:
    """Induce a subhypergraph from selected nodes and edges.
    
    Optimizations:
    - Input is assumed to already have unique, sorted IDs from expand_hyperedge_centered_subhypergraph
    - Avoid redundant sorted(set(...)) calls
    """
    # Convert to sorted list if not already (for consistency)
    ordered_nodes = sorted(set(int(node_id) for node_id in node_ids)) if node_ids else []
    ordered_edges = sorted(set(int(edge_id) for edge_id in edge_ids)) if edge_ids else []
    
    node_mapping = {global_id: local_id for local_id, global_id in enumerate(ordered_nodes)}
    local_hyperedges: List[List[int]] = []
    kept_edge_ids: List[int] = []
    
    for edge_id in ordered_edges:
        # Direct access, filter only nodes in node_mapping
        local_edge = [node_mapping[n] for n in hg.hyperedges[edge_id] if n in node_mapping]
        if local_edge:  # Only append non-empty edges
            local_edge.sort()
            local_hyperedges.append(local_edge)
            kept_edge_ids.append(edge_id)

    metadata = dict(hg.metadata)
    metadata.update(
        {
            "parent_graph_name": hg.name,
            "parent_dataset_name": hg.dataset_name,
            "global_node_ids": ordered_nodes,
            "global_edge_ids": kept_edge_ids,
            "seed_edge_ids": list(seed_edge_ids),
            "sampling_depth": sampling_depth,
            "sampling_strategy": "hyperedge_centered",
        }
    )
    # Efficient tensor indexing - advanced indexing already returns a new tensor
    # so .clone() is redundant. Using torch.tensor with dtype for efficiency.
    ordered_nodes_t = torch.tensor(ordered_nodes, dtype=torch.long)
    kept_edges_t = torch.tensor(kept_edge_ids, dtype=torch.long)
    
    return SimpleHypergraph(
        num_nodes=len(ordered_nodes),
        hyperedges=local_hyperedges,
        x=hg.x[ordered_nodes_t],
        name=f"{hg.name}_subhypergraph_{len(seed_edge_ids)}_{len(kept_edge_ids)}_{len(ordered_nodes)}",
        domain=hg.domain,
        dataset_name=hg.dataset_name,
        node_labels=hg.node_labels[ordered_nodes_t],
        edge_labels=hg.edge_labels[kept_edges_t] if hg.edge_labels is not None else None,
        graph_label=hg.graph_label if hg.graph_label is not None else None,  # Same for all subgraphs
        node_train_mask=hg.node_train_mask[ordered_nodes_t] if hg.node_train_mask is not None else None,
        node_val_mask=hg.node_val_mask[ordered_nodes_t] if hg.node_val_mask is not None else None,
        node_test_mask=hg.node_test_mask[ordered_nodes_t] if hg.node_test_mask is not None else None,
        metadata=metadata,
    )


def expand_hyperedge_centered_subhypergraph(
    hg: SimpleHypergraph,
    seed_edge_ids: Sequence[int],
    max_nodes: int,
    max_edges: int,
    expansion_hops: int,
    seed: int,
) -> SimpleHypergraph:
    """Sample a subhypergraph centered on seed edges with efficient expansion.
    
    Optimizations:
    - Pre-compute frozensets for all edges (O(num_edges × avg_edge_size) once)
    - Pre-compute node_to_edges adjacency (O(num_edges × avg_edge_size) once)
    - Use set operations efficiently
    """
    if not hg.hyperedges:
        return induce_sampled_subhypergraph(hg, [], [], [], sampling_depth=0)

    chosen_seed_edges = [edge_id for edge_id in seed_edge_ids if 0 <= edge_id < len(hg.hyperedges)]
    if not chosen_seed_edges:
        chosen_seed_edges = [0]
    selected_edges = list(dict.fromkeys(chosen_seed_edges))
    
    # Pre-compute edge frozensets for fast overlap computation
    edge_frozensets = [frozenset(edge) for edge in hg.hyperedges]
    edge_sets = edge_frozensets  # Alias for clarity
    
    selected_nodes = set()
    for edge_id in selected_edges:
        selected_nodes.update(edge_frozensets[edge_id])
    
    # Pre-compute node_to_edges adjacency
    node_to_edges: Dict[int, List[int]] = {}
    for edge_id, edge_set in enumerate(edge_frozensets):
        for node_id in edge_set:
            node_to_edges.setdefault(node_id, []).append(edge_id)

    for hop in range(expansion_hops):
        # Build frontier efficiently
        frontier = set()
        for node_id in selected_nodes:
            frontier.update(node_to_edges.get(node_id, []))
        
        candidate_edges = [e for e in frontier if e not in selected_edges]
        if not candidate_edges:
            break
            
        ranked_frontier = _rank_frontier_edges(
            hg,
            candidate_edges,
            selected_nodes,
            edge_sets=edge_sets,
            seed=seed + hop,
        )
        
        if not ranked_frontier:
            break
        added = False
        for edge_id in ranked_frontier:
            if len(selected_edges) >= max_edges:
                break
            edge_node_set = edge_frozensets[edge_id]
            new_nodes = edge_node_set - selected_nodes
            if len(selected_nodes) + len(new_nodes) > max_nodes:
                continue
            selected_edges.append(edge_id)
            selected_nodes.update(edge_node_set)
            added = True
            if len(selected_edges) >= max_edges or len(selected_nodes) >= max_nodes:
                break
        if not added:
            break

    return induce_sampled_subhypergraph(
        hg,
        node_ids=sorted(selected_nodes),
        edge_ids=selected_edges[:max_edges],
        seed_edge_ids=selected_edges[: len(chosen_seed_edges)],
        sampling_depth=expansion_hops,
    )


def sample_online_subhypergraph(hg: SimpleHypergraph, minibatch_config: Dict, seed: int) -> SimpleHypergraph:
    seed_edge_ids = sample_seed_hyperedges(
        hg,
        num_seeds=int(minibatch_config.get("seed_edges_per_subhypergraph", 1)),
        seed=seed,
    )
    return expand_hyperedge_centered_subhypergraph(
        hg,
        seed_edge_ids=seed_edge_ids,
        max_nodes=int(minibatch_config.get("max_nodes", 256)),
        max_edges=int(minibatch_config.get("max_edges", 128)),
        expansion_hops=int(minibatch_config.get("expansion_hops", 2)),
        seed=seed,
    )


def should_use_subhypergraph_pool(hg: SimpleHypergraph, minibatch_config: Dict) -> bool:
    return bool(minibatch_config.get("use_subhypergraph_pool", False)) and hg.num_nodes >= int(
        minibatch_config.get("large_graph_node_threshold", 5000)
    )


def build_subhypergraph_pool(hg: SimpleHypergraph, minibatch_config: Dict, seed: int) -> List[SimpleHypergraph]:
    pool_size = int(minibatch_config.get("subhypergraph_pool_size", 128))
    pool: List[SimpleHypergraph] = []
    for pool_index in range(pool_size):
        subhypergraph = sample_online_subhypergraph(hg, minibatch_config=minibatch_config, seed=seed + pool_index * 17)
        if subhypergraph.num_nodes == 0 or not subhypergraph.hyperedges:
            continue
        subhypergraph.name = f"{hg.name}_pool_{pool_index}"
        pool.append(subhypergraph)
    return pool


def sample_subhypergraph_batch(
    domains: Dict[str, List[SimpleHypergraph]],
    minibatch_config: Dict,
    pool_cache: Dict[str, List[SimpleHypergraph]],
    seed: int,
    preferred_domains: Sequence[str] | None = None,
) -> List[SimpleHypergraph]:
    """Sample a batch of sub-hypergraphs (legacy version)."""
    batch = sample_subhypergraph_batch_with_quality(
        domains=domains,
        minibatch_config=minibatch_config,
        pool_cache=pool_cache,
        seed=seed,
        preferred_domains=preferred_domains,
    )
    return [item["subhypergraph"] for item in batch]


def sample_subhypergraph_batch_with_quality(
    domains: Dict[str, List[SimpleHypergraph]],
    minibatch_config: Dict,
    pool_cache: Dict[str, List[SimpleHypergraph]],
    seed: int,
    preferred_domains: Sequence[str] | None = None,
) -> List[Dict]:
    """
    Sample a batch of sub-hypergraphs with quality metadata.
    
    Returns:
        List of dicts with keys:
        - subhypergraph: SimpleHypergraph
        - quality_meta: SubhypergraphMeta
    """
    requested = set(preferred_domains) if preferred_domains is not None else None
    available_domains = [domain for domain, graphs in domains.items() if graphs and (requested is None or domain in requested)]
    if not available_domains:
        return []
    generator = torch.Generator().manual_seed(seed)
    if requested is not None:
        chosen_domains = list(available_domains)
    else:
        domains_per_step = min(int(minibatch_config.get("domains_per_step", 2)), len(available_domains))
        domain_indices = torch.randperm(len(available_domains), generator=generator)[:domains_per_step].tolist()
        chosen_domains = [available_domains[index] for index in domain_indices]

    sampled: List[Dict] = []
    
    # Get quality routing thresholds from config
    tau_hyperedge = float(minibatch_config.get("tau_hyperedge", 0.55))
    tau_motif = float(minibatch_config.get("tau_motif", 0.70))
    tau_membership = float(minibatch_config.get("tau_membership", 0.40))
    tau_hard_negative = float(minibatch_config.get("tau_hard_negative", 0.40))
    min_membership_nodes = int(minibatch_config.get("min_membership_nodes", 2))
    min_nodes = int(minibatch_config.get("min_nodes", 4))
    min_edges = int(minibatch_config.get("min_edges", 2))
    
    for domain_offset, domain_name in enumerate(chosen_domains):
        graphs = domains[domain_name]
        domain_id = list(domains.keys()).index(domain_name)
        subhypergraphs_per_domain = int(minibatch_config.get("subhypergraphs_per_domain", 2))
        graph_indices = torch.randint(0, len(graphs), (subhypergraphs_per_domain,), generator=generator).tolist()
        for subhypergraph_offset, graph_index in enumerate(graph_indices):
            graph = graphs[graph_index]
            if should_use_subhypergraph_pool(graph, minibatch_config) and pool_cache.get(graph.name):
                pool = pool_cache[graph.name]
                pool_index = int(torch.randint(0, len(pool), (1,), generator=generator).item())
                subhg = pool[pool_index]
            else:
                local_seed = seed + domain_offset * 101 + subhypergraph_offset * 17 + graph_index
                subhg = sample_online_subhypergraph(graph, minibatch_config=minibatch_config, seed=local_seed)
            
            if subhg.num_nodes == 0 or not subhg.hyperedges:
                continue
            
            # Compute quality metadata
            quality_meta = compute_subhypergraph_quality(
                hg=subhg,
                min_nodes=min_nodes,
                min_edges=min_edges,
                tau_hyperedge=tau_hyperedge,
                tau_motif=tau_motif,
                tau_membership=tau_membership,
                tau_hard_negative=tau_hard_negative,
                domain_id=domain_id,
            )
            
            sampled.append({
                "subhypergraph": subhg,
                "quality_meta": quality_meta,
            })
    
    return sampled
