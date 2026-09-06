from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from v2.utils.dataset_registry import get_dataset_spec
from v2.utils.hypergraph import SimpleHypergraph


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


def _try_load_from_cache(dataset_name: str, target_dim: int, seed: int, data_root: Optional[str], require_node_splits: bool) -> Optional[SimpleHypergraph]:
    """Fallback loader: read edge_list.pkl / features.pkl / masks.pkl directly from
    data/cache to avoid the `dhg` module when the offline pickles already exist.

    Only covers the offline-pickled node_cls datasets in data/cache (rec datasets
    still go through dhg because their rec pickle layout is different).
    """
    if data_root is None:
        return None
    # Map logical names used in yaml → actual cache folder basenames.
    _alias = {
        "cora_cc": "cocitation_cora",
        "citeseer_cc": "cocitation_citeseer",
        "pubmed_cc": "cocitation_pubmed",
    }
    cache_name = _alias.get(dataset_name, dataset_name)
    cache_dir = Path(data_root) / cache_name
    if not cache_dir.is_dir():
        return None
    import pickle as _pkl
    def _load_pkl(name: str):
        p = cache_dir / name
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                return _pkl.load(f)
        except Exception:
            # Typical failure: the pickle was serialised with a scipy/numpy
            # version whose classes can't be imported now (scipy sparse, etc.).
            # Caller falls back to default/random features for feature.pkl and
            # to default empty masks/zeros-labels for others (safe; we only
            # absolutely require edge_list).
            return None
    # node_cls datasets need edge_list (mandatory).
    edge_list = _load_pkl("edge_list.pkl")
    if edge_list is None:
        # imdb_4k is stored as edge_by_actor + edge_by_director
        a = _load_pkl("edge_by_actor.pkl"); d = _load_pkl("edge_by_director.pkl")
        if a is not None or d is not None:
            merged = []
            for e in (a or []): merged.append(e)
            for e in (d or []): merged.append(e)
            edge_list = merged
        if edge_list is None:
            return None
    hyperedges = _normalize_edge_list(edge_list)
    labels = _load_pkl("labels.pkl")
    features = _load_pkl("features.pkl")
    train_mask = _load_pkl("train_mask.pkl")
    val_mask = _load_pkl("val_mask.pkl")
    test_mask = _load_pkl("test_mask.pkl")

    # Infer num_nodes: max node id across edges or feature rows / labels len
    num_nodes_hint = 0
    if hyperedges:
        for e in hyperedges:
            if e:
                num_nodes_hint = max(num_nodes_hint, int(max(e)) + 1)
    if features is not None and hasattr(features, "shape"):
        num_nodes_hint = max(num_nodes_hint, int(features.shape[0]))
    if labels is not None and hasattr(labels, "numel"):
        num_nodes_hint = max(num_nodes_hint, int(labels.numel()))
    num_nodes = int(num_nodes_hint)
    if num_nodes == 0:
        return None

    labels_t = torch.zeros((num_nodes,), dtype=torch.long)
    if labels is not None:
        try:
            labels_t = torch.as_tensor(labels, dtype=torch.long)
            if labels_t.numel() < num_nodes:
                labels_t = torch.cat([labels_t, torch.zeros(num_nodes - labels_t.numel(), dtype=torch.long)])
        except Exception:
            labels_t = torch.zeros((num_nodes,), dtype=torch.long)

    if features is None:
        x = _build_fallback_features(num_nodes, target_dim=target_dim, seed=seed)
    else:
        try:
            x = _resize_features(torch.as_tensor(features), target_dim=target_dim, seed=seed)
        except Exception:
            x = _build_fallback_features(num_nodes, target_dim=target_dim, seed=seed)

    def _to_bool_mask(v, size):
        if v is None:
            return None
        try:
            m = torch.as_tensor(v).bool()
            if m.numel() < size:
                m = torch.cat([m, torch.zeros(size - m.numel(), dtype=torch.bool)])
            return m[:size]
        except Exception:
            return None
    tr, va, te = (_to_bool_mask(m, num_nodes) for m in (train_mask, val_mask, test_mask))
    if require_node_splits and (tr is None or va is None or te is None):
        return None

    metadata = {
        **_derive_dataset_stats(num_nodes, hyperedges, x),
        "task_type": "node_cls",
        "num_node_classes": int(labels_t.max().item()) + 1 if labels_t.numel() else 0,
    }
    if tr is not None and va is not None and te is not None:
        metadata.update({
            "train_nodes": int(tr.sum().item()),
            "val_nodes": int(va.sum().item()),
            "test_nodes": int(te.sum().item()),
        })
    spec = get_dataset_spec(dataset_name)
    return SimpleHypergraph(
        num_nodes=num_nodes,
        hyperedges=hyperedges,
        x=torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
        name=dataset_name,
        domain=spec.domain,
        dataset_name=dataset_name,
        node_labels=labels_t.long(),
        edge_labels=None,
        graph_label=None,
        node_train_mask=tr,
        node_val_mask=va,
        node_test_mask=te,
        metadata=metadata,
    )


def _try_load_rec_from_cache(dataset_name: str, target_dim: int, seed: int, data_root: Optional[str]) -> Optional[SimpleHypergraph]:
    if data_root is None:
        return None
    cache_dir = Path(data_root) / dataset_name
    trp = cache_dir / "train.txt"
    tep = cache_dir / "test.txt"
    if not trp.is_file() or not tep.is_file():
        return None
    def _read_adj(path: Path):
        rows: Dict[int, List[int]] = {}
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                vals = [int(x) for x in parts]
                if len(vals) < 2:
                    continue
                uid = vals[0]
                items = vals[1:]
                rows[uid] = list(dict.fromkeys(items))
        return rows
    train_adj = _read_adj(trp)
    test_adj = _read_adj(tep)
    if not train_adj:
        return None
    num_users = max(max(train_adj.keys()), max(test_adj.keys())) + 1 if test_adj else max(train_adj.keys()) + 1
    num_items = 0
    for u, vs in train_adj.items():
        for v in vs:
            if v > num_items: num_items = v
    for u, vs in test_adj.items():
        for v in vs:
            if v > num_items: num_items = v
    num_items += 1
    spec = get_dataset_spec(dataset_name)
    train_items_by_user = []
    test_items_by_user = []
    hyperedges: List[List[int]] = []
    for uid in range(num_users):
        tr_items = train_adj.get(uid, [])
        te_items = test_adj.get(uid, [])
        train_items_by_user.append(tr_items)
        test_items_by_user.append(te_items)
        hyperedges.append(sorted(set(tr_items)) if tr_items else [])
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
        domain=spec.domain,
        dataset_name=dataset_name,
        node_labels=labels,
        edge_labels=None,
        graph_label=None,
        node_train_mask=None,
        node_val_mask=None,
        node_test_mask=None,
        metadata=metadata,
    )


def load_dhg_sample(
    dataset_name: str,
    target_dim: int,
    seed: int,
    data_root: Optional[str] = None,
    require_node_splits: bool = False,
) -> SimpleHypergraph:
    spec = get_dataset_spec(dataset_name)
    # Fast offline paths: skip dhg import if cache pickles/txt exist.
    if spec.task_type == "node_cls":
        cached = _try_load_from_cache(dataset_name, target_dim=target_dim, seed=seed,
                                      data_root=data_root, require_node_splits=require_node_splits)
        if cached is not None:
            return cached
    if spec.task_type == "rec":
        cached = _try_load_rec_from_cache(dataset_name, target_dim=target_dim, seed=seed,
                                          data_root=data_root)
        if cached is not None:
            return cached
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
