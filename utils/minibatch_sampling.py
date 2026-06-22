from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from utils.hypergraph import SimpleHypergraph


def sample_seed_hyperedges(hg: SimpleHypergraph, num_seeds: int, seed: int) -> List[int]:
    if not hg.hyperedges or num_seeds <= 0:
        return []
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(hg.hyperedges), generator=generator)
    return permutation[: min(num_seeds, len(hg.hyperedges))].tolist()


def _rank_frontier_edges(hg: SimpleHypergraph, candidate_edges: Sequence[int], selected_nodes: set[int], seed: int) -> List[int]:
    if not candidate_edges:
        return []
    generator = torch.Generator().manual_seed(seed)
    noise = torch.rand(len(candidate_edges), generator=generator).tolist()
    scored = []
    for offset, edge_id in enumerate(candidate_edges):
        overlap = len(selected_nodes.intersection(hg.hyperedges[edge_id]))
        scored.append((overlap, noise[offset], edge_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [edge_id for _, _, edge_id in scored]


def induce_sampled_subhypergraph(
    hg: SimpleHypergraph,
    node_ids: Sequence[int],
    edge_ids: Sequence[int],
    seed_edge_ids: Sequence[int],
    sampling_depth: int,
) -> SimpleHypergraph:
    ordered_nodes = sorted(set(int(node_id) for node_id in node_ids))
    ordered_edges = sorted(set(int(edge_id) for edge_id in edge_ids))
    node_mapping = {global_id: local_id for local_id, global_id in enumerate(ordered_nodes)}
    local_hyperedges: List[List[int]] = []
    kept_edge_ids: List[int] = []
    for edge_id in ordered_edges:
        local_edge = [node_mapping[node_id] for node_id in hg.hyperedges[edge_id] if node_id in node_mapping]
        if not local_edge:
            continue
        local_hyperedges.append(sorted(local_edge))
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
    return SimpleHypergraph(
        num_nodes=len(ordered_nodes),
        hyperedges=local_hyperedges,
        x=hg.x[ordered_nodes].clone(),
        name=f"{hg.name}_subhypergraph_{len(seed_edge_ids)}_{len(kept_edge_ids)}_{len(ordered_nodes)}",
        domain=hg.domain,
        dataset_name=hg.dataset_name,
        node_labels=hg.node_labels[ordered_nodes].clone(),
        edge_labels=hg.edge_labels[kept_edge_ids].clone() if hg.edge_labels is not None else None,
        graph_label=hg.graph_label.clone() if hg.graph_label is not None else None,
        node_train_mask=hg.node_train_mask[ordered_nodes].clone() if hg.node_train_mask is not None else None,
        node_val_mask=hg.node_val_mask[ordered_nodes].clone() if hg.node_val_mask is not None else None,
        node_test_mask=hg.node_test_mask[ordered_nodes].clone() if hg.node_test_mask is not None else None,
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
    if not hg.hyperedges:
        return induce_sampled_subhypergraph(hg, [], [], [], sampling_depth=0)

    chosen_seed_edges = [edge_id for edge_id in seed_edge_ids if 0 <= edge_id < len(hg.hyperedges)]
    if not chosen_seed_edges:
        chosen_seed_edges = [0]
    selected_edges = list(dict.fromkeys(chosen_seed_edges))
    selected_nodes = {node_id for edge_id in selected_edges for node_id in hg.hyperedges[edge_id]}
    node_to_edges: Dict[int, List[int]] = {}
    for edge_id, edge in enumerate(hg.hyperedges):
        for node_id in edge:
            node_to_edges.setdefault(node_id, []).append(edge_id)

    for hop in range(expansion_hops):
        frontier = set()
        for node_id in selected_nodes:
            frontier.update(node_to_edges.get(node_id, []))
        ranked_frontier = _rank_frontier_edges(
            hg,
            [edge_id for edge_id in frontier if edge_id not in selected_edges],
            selected_nodes,
            seed=seed + hop,
        )
        if not ranked_frontier:
            break
        added = False
        for edge_id in ranked_frontier:
            if len(selected_edges) >= max_edges:
                break
            candidate_nodes = set(hg.hyperedges[edge_id])
            new_nodes = candidate_nodes.difference(selected_nodes)
            if len(selected_nodes) + len(new_nodes) > max_nodes:
                continue
            selected_edges.append(edge_id)
            selected_nodes.update(candidate_nodes)
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

    sampled: List[SimpleHypergraph] = []
    for domain_offset, domain_name in enumerate(chosen_domains):
        graphs = domains[domain_name]
        subhypergraphs_per_domain = int(minibatch_config.get("subhypergraphs_per_domain", 2))
        graph_indices = torch.randint(0, len(graphs), (subhypergraphs_per_domain,), generator=generator).tolist()
        for subhypergraph_offset, graph_index in enumerate(graph_indices):
            graph = graphs[graph_index]
            if should_use_subhypergraph_pool(graph, minibatch_config) and pool_cache.get(graph.name):
                pool = pool_cache[graph.name]
                pool_index = int(torch.randint(0, len(pool), (1,), generator=generator).item())
                sampled.append(pool[pool_index])
                continue
            local_seed = seed + domain_offset * 101 + subhypergraph_offset * 17 + graph_index
            sampled.append(sample_online_subhypergraph(graph, minibatch_config=minibatch_config, seed=local_seed))
    return [subhypergraph for subhypergraph in sampled if subhypergraph.num_nodes > 0 and subhypergraph.hyperedges]
