from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from utils.dataset_registry import get_dataset_spec
from utils.hypergraph import SimpleHypergraph


def _content_keys(dataset) -> List[str]:
    try:
        return list(dataset.content.keys())
    except Exception:
        pass
    try:
        return list(dataset._content.keys())
    except Exception:
        return []


def _get_item(dataset, key: str):
    try:
        return dataset[key]
    except Exception:
        pass
    try:
        return getattr(dataset, "_content", {}).get(key)
    except Exception:
        return None


def _normalize_edge_list(edge_list) -> List[List[int]]:
    normalized: List[List[int]] = []
    for edge in edge_list:
        if isinstance(edge, set):
            items = list(edge)
        else:
            items = list(edge)
        normalized.append(sorted(int(node_id) for node_id in items))
    return normalized


def _extract_hyperedges(dataset_name: str, dataset) -> List[List[int]]:
    edge_list = _get_item(dataset, "edge_list")
    if edge_list is not None:
        return _normalize_edge_list(edge_list)
    # IMDB4k exposes two hyperedge families instead of a single edge_list.
    edge_by_actor = _get_item(dataset, "edge_by_actor")
    edge_by_director = _get_item(dataset, "edge_by_director")
    if edge_by_actor is not None or edge_by_director is not None:
        merged_edges = []
        for edges in (edge_by_actor, edge_by_director):
            if edges is None:
                continue
            merged_edges.extend(edges)
        return _normalize_edge_list(merged_edges)
    raise ValueError(f"Dataset '{dataset_name}' does not expose a supported hyperedge field.")


def _derive_dataset_stats(num_nodes: int, hyperedges: List[List[int]], x: torch.Tensor) -> Dict[str, float | int]:
    sizes = [len(edge) for edge in hyperedges if edge]
    if sizes:
        avg_size = float(sum(sizes) / len(sizes))
        max_size = int(max(sizes))
    else:
        avg_size = 0.0
        max_size = 0
    return {
        "num_nodes": int(num_nodes),
        "num_hyperedges": int(len(hyperedges)),
        "avg_hyperedge_size": float(avg_size),
        "max_hyperedge_size": int(max_size),
        "feature_dim": int(x.size(1)) if x is not None and x.ndim == 2 else 0,
    }


def _resolve_node_masks(dataset, num_nodes: int, require: bool) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    keys = set(_content_keys(dataset))
    has_masks = {"train_mask", "val_mask", "test_mask"}.issubset(keys)
    if not has_masks:
        if require:
            raise ValueError("Dataset does not provide official node splits (train/val/test masks).")
        return None, None, None
    return (
        torch.as_tensor(_get_item(dataset, "train_mask")).bool(),
        torch.as_tensor(_get_item(dataset, "val_mask")).bool(),
        torch.as_tensor(_get_item(dataset, "test_mask")).bool(),
    )


def _resize_features(features: torch.Tensor, target_dim: int, seed: int) -> torch.Tensor:
    features = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if features.ndim != 2:
        raise ValueError("Expected 2D node feature matrix.")
    if features.size(1) == target_dim:
        return features
    if features.size(1) > target_dim:
        generator = torch.Generator().manual_seed(seed)
        projection = torch.randn(features.size(1), target_dim, generator=generator, dtype=features.dtype)
        return features @ projection / max(target_dim, 1) ** 0.5
    padding = features.new_zeros((features.size(0), target_dim - features.size(1)))
    return torch.cat([features, padding], dim=1)


def _build_fallback_features(num_nodes: int, target_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num_nodes, target_dim, generator=generator)


def load_dhg_sample(
    dataset_name: str,
    target_dim: int,
    seed: int,
    data_root: Optional[str] = None,
    require_node_splits: bool = False,
) -> SimpleHypergraph:
    spec = get_dataset_spec(dataset_name)
    root = str(Path(data_root)) if data_root else None
    dataset = spec.loader(root)
    keys = set(_content_keys(dataset))

    if spec.task_type == "rec":
        return _load_recommendation_dataset(dataset_name, spec.domain, dataset, target_dim=target_dim, seed=seed)

    raw_labels = _get_item(dataset, "labels")
    if raw_labels is None:
        labels = torch.zeros((int(_get_item(dataset, "num_vertices") or 0),), dtype=torch.long)
    else:
        labels = torch.as_tensor(raw_labels, dtype=torch.long)

    num_nodes = int(_get_item(dataset, "num_vertices") or labels.numel())
    if "features" in keys:
        features = torch.as_tensor(_get_item(dataset, "features"))
        x = _resize_features(features, target_dim=target_dim, seed=seed)
    else:
        x = _build_fallback_features(num_nodes, target_dim=target_dim, seed=seed)

    train_mask, val_mask, test_mask = _resolve_node_masks(dataset, num_nodes=num_nodes, require=require_node_splits)
    hyperedges = _extract_hyperedges(dataset_name, dataset)
    metadata = {
        **_derive_dataset_stats(num_nodes, hyperedges, x),
        "task_type": spec.task_type,
        "num_node_classes": int(labels.max().item()) + 1 if labels.numel() else 0,
    }
    if train_mask is not None and val_mask is not None and test_mask is not None:
        metadata.update(
            {
                "train_nodes": int(train_mask.sum().item()),
                "val_nodes": int(val_mask.sum().item()),
                "test_nodes": int(test_mask.sum().item()),
            }
        )

    return SimpleHypergraph(
        num_nodes=num_nodes,
        hyperedges=hyperedges,
        x=torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
        name=dataset_name,
        domain=spec.domain,
        dataset_name=dataset_name,
        node_labels=labels.long(),
        edge_labels=None,
        graph_label=None,
        node_train_mask=train_mask,
        node_val_mask=val_mask,
        node_test_mask=test_mask,
        metadata=metadata,
    )


def _load_recommendation_dataset(dataset_name: str, domain: str, dataset, target_dim: int, seed: int) -> SimpleHypergraph:
    num_items = int(_get_item(dataset, "num_items") or 0)
    num_users = int(_get_item(dataset, "num_users") or 0)
    train_adj = _get_item(dataset, "train_adj_list")
    test_adj = _get_item(dataset, "test_adj_list")
    if train_adj is None or test_adj is None:
        raise ValueError(f"Recommendation dataset '{dataset_name}' does not expose train/test adjacency lists.")
    if num_items <= 0 or num_users <= 0:
        raise ValueError(f"Recommendation dataset '{dataset_name}' does not expose num_users/num_items.")

    hyperedges: List[List[int]] = []
    train_items_by_user: List[List[int]] = []
    test_items_by_user: List[List[int]] = []
    for user_id in range(num_users):
        train_items = _normalize_recommendation_row(train_adj[user_id], user_id=user_id, num_items=num_items)
        test_items = _normalize_recommendation_row(test_adj[user_id], user_id=user_id, num_items=num_items)
        train_items_by_user.append(train_items)
        test_items_by_user.append(test_items)
        if train_items:
            hyperedges.append(sorted(set(train_items)))
        else:
            hyperedges.append([])

    x = _build_fallback_features(num_items, target_dim=target_dim, seed=seed)
    labels = torch.zeros((num_items,), dtype=torch.long)

    metadata = {
        **_derive_dataset_stats(num_items, hyperedges, x),
        "task_type": "rec",
        "num_users": int(num_users),
        "num_items": int(num_items),
        "train_adj_list": train_items_by_user,
        "test_adj_list": test_items_by_user,
    }

    return SimpleHypergraph(
        num_nodes=num_items,
        hyperedges=hyperedges,
        x=torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
        name=dataset_name,
        domain=domain,
        dataset_name=dataset_name,
        node_labels=labels,
        edge_labels=None,
        graph_label=None,
        node_train_mask=None,
        node_val_mask=None,
        node_test_mask=None,
        metadata=metadata,
    )


def _normalize_recommendation_row(row, user_id: int, num_items: int) -> List[int]:
    values = [int(item_id) for item_id in row]
    if values and values[0] == user_id:
        values = values[1:]
    normalized = [item_id for item_id in values if 0 <= item_id < num_items]
    return list(dict.fromkeys(normalized))


def load_domain_graphs(config: Dict, seed: int, require_node_splits: bool = False) -> Dict[str, List[SimpleHypergraph]]:
    data_config = config["data"]
    model_config = config["model"]
    domain_map = data_config.get("domain_map", {})
    graphs_by_domain: Dict[str, List[SimpleHypergraph]] = {}
    for dataset_name in data_config["datasets"]:
        graph = load_dhg_sample(
            dataset_name=dataset_name,
            target_dim=int(model_config["input_dim"]),
            seed=seed,
            data_root=data_config.get("cache_dir"),
            require_node_splits=require_node_splits,
        )
        if dataset_name in domain_map:
            graph.domain = domain_map[dataset_name]
        graphs_by_domain.setdefault(graph.domain, []).append(graph)
    return graphs_by_domain
