from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import sparse, stats
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v2.utils.dhg_datasets import load_dhg_sample
from v2.utils.dataset_registry import get_dataset_spec


@dataclass
class ProcessedHypergraph:
    name: str
    domain: str
    num_nodes: int
    hyperedges: List[Tuple[int, ...]]
    original_num_nodes: int
    original_num_hyperedges: int
    removed_singletons: int
    removed_duplicates: int
    removed_isolated_nodes: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def preprocess_graph(name: str, domain: str, num_nodes: int, hyperedges: Sequence[Sequence[int]]) -> ProcessedHypergraph:
    cleaned: List[Tuple[int, ...]] = []
    seen = set()
    removed_singletons = 0
    removed_duplicates = 0

    for edge in hyperedges:
        uniq = tuple(sorted({int(v) for v in edge}))
        if len(uniq) < 2:
            removed_singletons += 1
            continue
        if uniq in seen:
            removed_duplicates += 1
            continue
        seen.add(uniq)
        cleaned.append(uniq)

    active_nodes = sorted({v for edge in cleaned for v in edge})
    mapping = {old: new for new, old in enumerate(active_nodes)}
    reindexed = [tuple(mapping[v] for v in edge) for edge in cleaned]

    return ProcessedHypergraph(
        name=name,
        domain=domain,
        num_nodes=len(active_nodes),
        hyperedges=reindexed,
        original_num_nodes=int(num_nodes),
        original_num_hyperedges=int(len(hyperedges)),
        removed_singletons=int(removed_singletons),
        removed_duplicates=int(removed_duplicates),
        removed_isolated_nodes=int(num_nodes - len(active_nodes)),
    )


def incidence_matrix(graph: ProcessedHypergraph) -> sparse.csr_matrix:
    rows: List[int] = []
    cols: List[int] = []
    for e_idx, edge in enumerate(graph.hyperedges):
        rows.extend(edge)
        cols.extend([e_idx] * len(edge))
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(graph.num_nodes, len(graph.hyperedges)))


def build_bipartite_adjacency(B: sparse.csr_matrix) -> sparse.csr_matrix:
    n_nodes, n_edges = B.shape
    zero_v = sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float32)
    zero_e = sparse.csr_matrix((n_edges, n_edges), dtype=np.float32)
    return sparse.bmat([[zero_v, B], [B.T, zero_e]], format="csr")


def gini_coefficient(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(np.maximum(x, 0.0))
    n = x.size
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n)


def safe_skew(x: np.ndarray) -> float:
    if x.size < 3 or np.allclose(x, x[0]):
        return 0.0
    return float(stats.skew(x, bias=False))


def js_similarity(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    dist = float(stats.entropy((p + q) / 2, base=2) - 0.5 * stats.entropy(p, base=2) - 0.5 * stats.entropy(q, base=2))
    dist = max(0.0, dist)
    return float(max(0.0, 1.0 - math.sqrt(dist)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def histogram(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins)
    return hist.astype(np.float64)


def bootstrap_mean_diff(a: Sequence[float], b: Sequence[float], seed: int, n_boot: int = 2000) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return (float("nan"), float("nan"))
    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=a.size, replace=True)
        sb = rng.choice(b, size=b.size, replace=True)
        diffs.append(float(np.mean(sa) - np.mean(sb)))
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = 0
    lt = 0
    for x in a:
        gt += int(np.sum(x > b))
        lt += int(np.sum(x < b))
    return float((gt - lt) / (a.size * b.size))


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return float("nan")
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled = math.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / max(a.size + b.size - 2, 1))
    if pooled == 0.0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def mann_whitney_pvalue(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def pair_labels(graphs: Sequence[ProcessedHypergraph]) -> List[Tuple[str, str, str]]:
    out = []
    for g1, g2 in combinations(graphs, 2):
        relation = "within" if g1.domain == g2.domain else "cross"
        out.append((g1.name, g2.name, relation))
    return out


def sample_overlap_features(graph: ProcessedHypergraph, pairs_per_repeat: int, repeats: int, seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    m = len(graph.hyperedges)
    edge_sets = [set(edge) for edge in graph.hyperedges]
    if m < 2:
        zeros = np.zeros(16, dtype=np.float64)
        return {
            "jaccard_hist": zeros,
            "norm_hist": zeros,
            "intersection_hist": zeros,
            "mean_jaccard": 0.0,
            "mean_norm_overlap": 0.0,
            "mean_intersection": 0.0,
            "share_any_ratio": 0.0,
            "sample_pairs": 0,
        }

    norm_bins = np.linspace(0.0, 1.0, 17)
    jacc_bins = np.linspace(0.0, 1.0, 17)
    max_size = max(len(edge) for edge in graph.hyperedges)
    int_bins = np.arange(0, max_size + 2) - 0.5

    jacc_hists = []
    norm_hists = []
    int_hists = []
    means_j = []
    means_n = []
    means_i = []
    share_any = []

    total_pairs = m * (m - 1) // 2
    sample_size = min(pairs_per_repeat, total_pairs)
    all_pairs = None
    if total_pairs <= sample_size:
        all_pairs = np.array(list(combinations(range(m), 2)), dtype=np.int64)

    for _ in range(repeats):
        if all_pairs is not None:
            sampled = all_pairs
        else:
            left = rng.integers(0, m, size=sample_size)
            right = rng.integers(0, m, size=sample_size)
            mask = left != right
            left = left[mask]
            right = right[mask]
            if left.size == 0:
                sampled = np.empty((0, 2), dtype=np.int64)
            else:
                a = np.minimum(left, right)
                b = np.maximum(left, right)
                sampled = np.unique(np.stack([a, b], axis=1), axis=0)
        inter_vals = []
        jacc_vals = []
        norm_vals = []
        for i, j in sampled:
            s1 = edge_sets[int(i)]
            s2 = edge_sets[int(j)]
            inter = len(s1 & s2)
            union = len(s1 | s2)
            denom = min(len(s1), len(s2))
            inter_vals.append(inter)
            jacc_vals.append(inter / union if union > 0 else 0.0)
            norm_vals.append(inter / denom if denom > 0 else 0.0)
        inter_arr = np.asarray(inter_vals, dtype=np.float64)
        jacc_arr = np.asarray(jacc_vals, dtype=np.float64)
        norm_arr = np.asarray(norm_vals, dtype=np.float64)
        jacc_hists.append(histogram(jacc_arr, jacc_bins))
        norm_hists.append(histogram(norm_arr, norm_bins))
        int_hists.append(histogram(inter_arr, int_bins))
        means_j.append(float(jacc_arr.mean()) if jacc_arr.size else 0.0)
        means_n.append(float(norm_arr.mean()) if norm_arr.size else 0.0)
        means_i.append(float(inter_arr.mean()) if inter_arr.size else 0.0)
        share_any.append(float(np.mean(inter_arr >= 1)) if inter_arr.size else 0.0)

    return {
        "jaccard_hist": np.mean(jacc_hists, axis=0),
        "norm_hist": np.mean(norm_hists, axis=0),
        "intersection_hist": np.mean(int_hists, axis=0),
        "mean_jaccard": float(np.mean(means_j)),
        "mean_norm_overlap": float(np.mean(means_n)),
        "mean_intersection": float(np.mean(means_i)),
        "share_any_ratio": float(np.mean(share_any)),
        "sample_pairs": int(sample_size),
    }


def size_features(graph: ProcessedHypergraph) -> Dict[str, object]:
    sizes = np.asarray([len(edge) for edge in graph.hyperedges], dtype=np.int64)
    max_size = int(sizes.max()) if sizes.size else 1
    bins = np.arange(1, max_size + 2) - 0.5
    return {
        "sizes": sizes,
        "pmf": histogram(sizes, bins),
        "mean": float(sizes.mean()) if sizes.size else 0.0,
        "median": float(np.median(sizes)) if sizes.size else 0.0,
        "variance": float(sizes.var()) if sizes.size else 0.0,
        "skewness": safe_skew(sizes.astype(np.float64)),
        "tail_p90": float(np.percentile(sizes, 90)) if sizes.size else 0.0,
        "tail_p95": float(np.percentile(sizes, 95)) if sizes.size else 0.0,
    }


def incidence_features(graph: ProcessedHypergraph) -> Dict[str, object]:
    B = incidence_matrix(graph)
    node_deg = np.asarray(B.sum(axis=1)).ravel().astype(np.float64)
    edge_sizes = np.asarray(B.sum(axis=0)).ravel().astype(np.float64)
    bip = build_bipartite_adjacency(B)
    n_components, _ = sparse.csgraph.connected_components(bip, directed=False)

    k = min(20, max(1, min(B.shape) - 1))
    if k >= 1:
        try:
            svals = svds(B.astype(np.float64), k=k, return_singular_vectors=False)
            svals = np.sort(svals)[::-1]
        except Exception:
            dense = B.toarray().astype(np.float64)
            svals = np.linalg.svd(dense, compute_uv=False)[:k]
    else:
        svals = np.zeros(1, dtype=np.float64)
    if svals.size > 0 and svals[0] > 0:
        svals = svals / svals[0]
    if svals.size < 20:
        svals = np.pad(svals, (0, 20 - svals.size))

    deg_bins = np.arange(0, int(max(node_deg.max(initial=0), 1)) + 2) - 0.5
    return {
        "B": B,
        "node_degree": node_deg,
        "edge_size": edge_sizes,
        "degree_hist": histogram(node_deg, deg_bins),
        "degree_mean": float(node_deg.mean()) if node_deg.size else 0.0,
        "degree_max": float(node_deg.max()) if node_deg.size else 0.0,
        "degree_gini": gini_coefficient(node_deg),
        "density": float(B.nnz / max(graph.num_nodes * len(graph.hyperedges), 1)),
        "num_connected_components": int(n_components),
        "spectrum": svals.astype(np.float64),
    }


def community_features(graph: ProcessedHypergraph, B: sparse.csr_matrix, seed: int) -> Dict[str, object]:
    G = nx.Graph()
    offset = graph.num_nodes
    G.add_nodes_from(range(graph.num_nodes), bipartite=0)
    G.add_nodes_from(range(offset, offset + len(graph.hyperedges)), bipartite=1)
    coo = B.tocoo()
    G.add_edges_from((int(r), offset + int(c)) for r, c in zip(coo.row, coo.col))

    communities = nx.algorithms.community.louvain_communities(G, seed=seed)
    sizes = np.asarray([len(c) for c in communities], dtype=np.float64)
    modularity = nx.algorithms.community.modularity(G, communities) if communities else 0.0
    bins = np.arange(1, int(max(sizes.max(initial=1), 1)) + 2) - 0.5
    probs = sizes / sizes.sum() if sizes.sum() > 0 else np.array([1.0])
    entropy = float(stats.entropy(probs))
    return {
        "num_communities": int(len(communities)),
        "community_sizes": sizes,
        "size_hist": histogram(sizes, bins),
        "modularity": float(modularity),
        "largest_ratio": float(sizes.max() / G.number_of_nodes()) if sizes.size and G.number_of_nodes() else 0.0,
        "entropy": entropy,
    }


def randomize_hyperedges(graph: ProcessedHypergraph, seed: int) -> ProcessedHypergraph:
    rng = np.random.default_rng(seed)
    hyperedges = []
    for edge in graph.hyperedges:
        size = len(edge)
        if size >= graph.num_nodes:
            nodes = tuple(range(graph.num_nodes))
        else:
            nodes = tuple(sorted(rng.choice(graph.num_nodes, size=size, replace=False).tolist()))
        hyperedges.append(nodes)
    return preprocess_graph(
        name=f"{graph.name}__random",
        domain=graph.domain,
        num_nodes=graph.num_nodes,
        hyperedges=hyperedges,
    )


def normalize_hist_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = max(len(a), len(b))
    aa = np.pad(np.asarray(a, dtype=np.float64), (0, n - len(a)))
    bb = np.pad(np.asarray(b, dtype=np.float64), (0, n - len(b)))
    return aa, bb


def compute_similarity_matrices(
    graphs: Sequence[ProcessedHypergraph],
    size_map: Dict[str, Dict[str, object]],
    overlap_map: Dict[str, Dict[str, object]],
    incidence_map: Dict[str, Dict[str, object]],
    community_map: Dict[str, Dict[str, object]],
) -> Dict[str, pd.DataFrame]:
    names = [g.name for g in graphs]
    index = pd.Index(names, name="dataset")

    size_mat = pd.DataFrame(np.eye(len(names)), index=index, columns=names)
    overlap_mat = pd.DataFrame(np.eye(len(names)), index=index, columns=names)
    incidence_mat = pd.DataFrame(np.eye(len(names)), index=index, columns=names)
    community_mat = pd.DataFrame(np.eye(len(names)), index=index, columns=names)

    global_size_range = max(
        max(int(size_map[g.name]["sizes"].max(initial=1)) for g in graphs) - 1,
        1,
    )

    for g1, g2 in combinations(graphs, 2):
        n1, n2 = g1.name, g2.name
        p1, p2 = normalize_hist_pair(size_map[n1]["pmf"], size_map[n2]["pmf"])
        js = js_similarity(p1, p2)
        wd = stats.wasserstein_distance(size_map[n1]["sizes"], size_map[n2]["sizes"])
        wsim = math.exp(-float(wd) / global_size_range)
        size_sim = 0.5 * js + 0.5 * wsim

        oj1, oj2 = normalize_hist_pair(overlap_map[n1]["jaccard_hist"], overlap_map[n2]["jaccard_hist"])
        on1, on2 = normalize_hist_pair(overlap_map[n1]["norm_hist"], overlap_map[n2]["norm_hist"])
        overlap_sim = 0.5 * js_similarity(oj1, oj2) + 0.5 * js_similarity(on1, on2)

        id1, id2 = normalize_hist_pair(incidence_map[n1]["degree_hist"], incidence_map[n2]["degree_hist"])
        incidence_sim = 0.5 * js_similarity(id1, id2) + 0.5 * cosine_similarity(incidence_map[n1]["spectrum"], incidence_map[n2]["spectrum"])

        cs1, cs2 = normalize_hist_pair(community_map[n1]["size_hist"], community_map[n2]["size_hist"])
        mod_gap = abs(community_map[n1]["modularity"] - community_map[n2]["modularity"])
        community_sim = 0.7 * js_similarity(cs1, cs2) + 0.3 * math.exp(-mod_gap)

        for mat, val in (
            (size_mat, size_sim),
            (overlap_mat, overlap_sim),
            (incidence_mat, incidence_sim),
            (community_mat, community_sim),
        ):
            mat.loc[n1, n2] = val
            mat.loc[n2, n1] = val

    return {
        "size": size_mat,
        "overlap": overlap_mat,
        "incidence": incidence_mat,
        "community": community_mat,
    }


def summarise_pairs(graphs: Sequence[ProcessedHypergraph], matrix: pd.DataFrame, random_mean: float, seed: int) -> Dict[str, float]:
    within_vals = []
    cross_vals = []
    for g1, g2 in combinations(graphs, 2):
        val = float(matrix.loc[g1.name, g2.name])
        if g1.domain == g2.domain:
            within_vals.append(val)
        else:
            cross_vals.append(val)
    mu_within = float(np.mean(within_vals)) if within_vals else float("nan")
    mu_cross = float(np.mean(cross_vals)) if cross_vals else float("nan")
    ci_low, ci_high = bootstrap_mean_diff(within_vals, cross_vals, seed=seed)
    eps = 1e-8
    tf = float((mu_cross - random_mean) / (mu_within - random_mean + eps)) if not math.isnan(mu_within) and not math.isnan(mu_cross) else float("nan")
    return {
        "mu_within": mu_within,
        "mu_cross": mu_cross,
        "mu_random": random_mean,
        "delta_within_minus_cross": mu_within - mu_cross if not math.isnan(mu_within) and not math.isnan(mu_cross) else float("nan"),
        "mannwhitney_p": mann_whitney_pvalue(within_vals, cross_vals),
        "cliffs_delta": cliffs_delta(within_vals, cross_vals),
        "cohens_d": cohens_d(within_vals, cross_vals),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "transferability_score": tf,
        "n_within_pairs": len(within_vals),
        "n_cross_pairs": len(cross_vals),
    }


def save_heatmap(df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7.5, 6.5))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", vmin=0.0, vmax=1.0, square=True, cbar_kws={"shrink": 0.8})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_size_distribution(graphs: Sequence[ProcessedHypergraph], size_map: Dict[str, Dict[str, object]], out_path: Path) -> None:
    plt.figure(figsize=(8, 5.5))
    for graph in graphs:
        sizes = np.asarray(size_map[graph.name]["sizes"], dtype=np.float64)
        xs = np.sort(np.unique(sizes))
        if xs.size == 0:
            continue
        cdf = np.array([np.mean(sizes <= x) for x in xs])
        plt.step(xs, cdf, where="post", label=graph.name)
    plt.xlabel("Hyperedge size")
    plt.ylabel("CDF")
    plt.title("Hyperedge Size Distribution")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_fingerprint_table(
    graphs: Sequence[ProcessedHypergraph],
    size_map: Dict[str, Dict[str, object]],
    overlap_map: Dict[str, Dict[str, object]],
    incidence_map: Dict[str, Dict[str, object]],
    community_map: Dict[str, Dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for g in graphs:
        rows.append(
            {
                "dataset": g.name,
                "domain": g.domain,
                "size_mean": size_map[g.name]["mean"],
                "size_median": size_map[g.name]["median"],
                "size_var": size_map[g.name]["variance"],
                "size_skew": size_map[g.name]["skewness"],
                "tail_p90": size_map[g.name]["tail_p90"],
                "overlap_mean_jaccard": overlap_map[g.name]["mean_jaccard"],
                "overlap_mean_norm": overlap_map[g.name]["mean_norm_overlap"],
                "overlap_share_any": overlap_map[g.name]["share_any_ratio"],
                "overlap_mean_intersection": overlap_map[g.name]["mean_intersection"],
                "degree_mean": incidence_map[g.name]["degree_mean"],
                "degree_max": incidence_map[g.name]["degree_max"],
                "degree_gini": incidence_map[g.name]["degree_gini"],
                "incidence_density": incidence_map[g.name]["density"],
                "num_components": incidence_map[g.name]["num_connected_components"],
                "community_count": community_map[g.name]["num_communities"],
                "community_modularity": community_map[g.name]["modularity"],
                "community_largest_ratio": community_map[g.name]["largest_ratio"],
                "community_entropy": community_map[g.name]["entropy"],
            }
        )
    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c not in {"dataset", "domain"}]
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].to_numpy(dtype=np.float64))
    n_components = min(3, max(1, X.shape[0] - 1), X.shape[1])
    if n_components >= 1:
        pca = PCA(n_components=n_components, whiten=False, random_state=0)
        Z = pca.fit_transform(X)
    else:
        Z = X
    fp_cols = [f"fp_{i+1}" for i in range(Z.shape[1])]
    fp_df = pd.concat([df[["dataset", "domain"]].reset_index(drop=True), pd.DataFrame(Z, columns=fp_cols)], axis=1)
    return fp_df


def fingerprint_similarity_matrix(fingerprint_df: pd.DataFrame) -> pd.DataFrame:
    names = fingerprint_df["dataset"].tolist()
    feats = fingerprint_df[[c for c in fingerprint_df.columns if c.startswith("fp_")]].to_numpy(dtype=np.float64)
    mat = np.zeros((len(names), len(names)), dtype=np.float64)
    for i in range(len(names)):
        for j in range(len(names)):
            mat[i, j] = cosine_similarity(feats[i], feats[j]) if i != j else 1.0
    return pd.DataFrame(mat, index=names, columns=names)


def basic_stats_row(graph: ProcessedHypergraph, incidence_map: Dict[str, object]) -> Dict[str, object]:
    sizes = np.asarray([len(edge) for edge in graph.hyperedges], dtype=np.float64)
    node_deg = np.asarray(incidence_map["node_degree"], dtype=np.float64)
    return {
        "dataset": graph.name,
        "domain": graph.domain,
        "num_nodes": graph.num_nodes,
        "num_hyperedges": len(graph.hyperedges),
        "num_unique_hyperedges": len(graph.hyperedges),
        "avg_hyperedge_size": float(sizes.mean()) if sizes.size else 0.0,
        "median_hyperedge_size": float(np.median(sizes)) if sizes.size else 0.0,
        "max_hyperedge_size": int(sizes.max()) if sizes.size else 0,
        "min_hyperedge_size": int(sizes.min()) if sizes.size else 0,
        "density": float(incidence_map["density"]),
        "avg_hyperdegree": float(node_deg.mean()) if node_deg.size else 0.0,
        "max_hyperdegree": float(node_deg.max()) if node_deg.size else 0.0,
        "num_connected_components": int(incidence_map["num_connected_components"]),
        "removed_singletons": graph.removed_singletons,
        "removed_duplicates": graph.removed_duplicates,
        "removed_isolated_nodes": graph.removed_isolated_nodes,
        "original_num_nodes": graph.original_num_nodes,
        "original_num_hyperedges": graph.original_num_hyperedges,
    }


def load_graphs(dataset_names: Sequence[str], target_dim: int, cache_dir: str, seed: int) -> List[ProcessedHypergraph]:
    graphs = []
    for name in dataset_names:
        spec = get_dataset_spec(name)
        raw = load_dhg_sample(name, target_dim=target_dim, seed=seed, data_root=cache_dir, require_node_splits=False)
        graphs.append(preprocess_graph(name=name, domain=spec.domain, num_nodes=raw.num_nodes, hyperedges=raw.hyperedges))
    return graphs


def resolve_device(requested: str) -> Tuple[str, List[str]]:
    notes = []
    if requested.startswith("cuda"):
        if os.environ.get("ALLOW_CUDA_PROBE", "0") == "1" and torch.cuda.is_available():
            return requested, notes
        notes.append("当前脚本默认回退到 CPU。若要真正绑定 GPU 7，请在非沙箱环境下设置 `CUDA_VISIBLE_DEVICES=7 ALLOW_CUDA_PROBE=1` 后重跑。")
    return "cpu", notes


def write_summary(
    out_path: Path,
    graphs: Sequence[ProcessedHypergraph],
    factor_summary: pd.DataFrame,
    fingerprint_df: pd.DataFrame,
    runtime_s: float,
    device: str,
    notes: Sequence[str],
) -> None:
    lines = [
        "# Cross-domain Hypergraph Transferability Summary",
        "",
        "## Run Setup",
        "",
        f"- Datasets: {', '.join(g.name for g in graphs)}",
        f"- Domains: {', '.join(sorted({g.domain for g in graphs}))}",
        f"- Runtime device: `{device}`",
        f"- Runtime: {runtime_s:.2f}s",
        "",
        "## Structural Factor Summary",
        "",
        factor_summary.to_markdown(index=False),
        "",
        "## Fingerprint PCA Coordinates",
        "",
        fingerprint_df.to_markdown(index=False),
        "",
        "## Open Questions / Current Gaps",
        "",
        "- `Lee et al.` 的标准 26 维 h-motif Characteristic Profile 本仓库内没有现成实现，因此本轮没有把 motif 纳入正式结果。",
        "- 文档推荐的 11 个 benchmark 数据集目前本地未齐备；本轮按最低交付标准先使用 6 个可直接读取的数据集。",
        "- Community 这里只跑了 incidence bipartite Louvain；clique expansion 版本对大超边可能显著放大边数，需单独设预算与近似策略。",
        "- recommendation 域当前使用的是 `train.txt/test.txt` 还原出的 item-hypergraph，语义上与纯 node classification 超图不同，后续需要确认是否保留在主表。",
    ]
    for note in notes:
        lines.append(f"- {note}")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="cora_cc,citeseer_cc,coauthorship_cora,coauthorship_dblp,cooking_200,gowalla")
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--target_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pairs_per_repeat", type=int, default=100000)
    parser.add_argument("--overlap_repeats", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="outputs_transferability/phase1_phase2_seed7")
    args = parser.parse_args()

    t0 = time.perf_counter()
    set_seed(args.seed)
    dataset_names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    out_dir = ROOT / args.output_dir
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    logs_dir = out_dir / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    device, device_notes = resolve_device(args.device)
    graphs = load_graphs(dataset_names, target_dim=args.target_dim, cache_dir=args.cache_dir, seed=args.seed)

    size_map = {g.name: size_features(g) for g in graphs}
    overlap_map = {
        g.name: sample_overlap_features(g, pairs_per_repeat=args.pairs_per_repeat, repeats=args.overlap_repeats, seed=args.seed + i * 17)
        for i, g in enumerate(graphs)
    }
    incidence_map = {g.name: incidence_features(g) for g in graphs}
    community_map = {
        g.name: community_features(g, incidence_map[g.name]["B"], seed=args.seed)
        for g in graphs
    }

    basic_stats = pd.DataFrame([basic_stats_row(g, incidence_map[g.name]) for g in graphs])
    basic_stats.to_csv(tables_dir / "dataset_basic_stats.csv", index=False)

    matrices = compute_similarity_matrices(graphs, size_map, overlap_map, incidence_map, community_map)
    for name, df in matrices.items():
        df.to_csv(tables_dir / f"{name}_similarity_matrix.csv")
        save_heatmap(df, figures_dir / f"{name}_similarity_heatmap.png", f"{name.capitalize()} Similarity")
    save_size_distribution(graphs, size_map, figures_dir / "hyperedge_size_distribution.png")

    random_graphs = [randomize_hyperedges(g, seed=args.seed + 1000 + idx) for idx, g in enumerate(graphs)]
    rand_size = {g.name: size_features(g) for g in random_graphs}
    rand_overlap = {
        g.name: sample_overlap_features(g, pairs_per_repeat=args.pairs_per_repeat, repeats=1, seed=args.seed + 2000 + i)
        for i, g in enumerate(random_graphs)
    }
    rand_incidence = {g.name: incidence_features(g) for g in random_graphs}
    rand_community = {
        g.name: community_features(g, rand_incidence[g.name]["B"], seed=args.seed)
        for g in random_graphs
    }
    random_mats = compute_similarity_matrices(random_graphs, rand_size, rand_overlap, rand_incidence, rand_community)

    factor_rows = []
    for factor, df in matrices.items():
        random_mean = float(np.mean(random_mats[factor].to_numpy()[np.triu_indices(len(random_graphs), k=1)]))
        row = {"factor": factor}
        row.update(summarise_pairs(graphs, df, random_mean=random_mean, seed=args.seed))
        factor_rows.append(row)
    factor_summary = pd.DataFrame(factor_rows)
    factor_summary.to_csv(tables_dir / "factor_transferability_summary.csv", index=False)

    fingerprint_df = build_fingerprint_table(graphs, size_map, overlap_map, incidence_map, community_map)
    fingerprint_df.to_csv(tables_dir / "structural_fingerprint_pca.csv", index=False)
    fp_matrix = fingerprint_similarity_matrix(fingerprint_df)
    fp_matrix.to_csv(tables_dir / "structural_fingerprint_similarity_matrix.csv")
    save_heatmap(fp_matrix, figures_dir / "structural_fingerprint_similarity_heatmap.png", "Structural Fingerprint Similarity")

    run_meta = {
        "datasets": dataset_names,
        "seed": args.seed,
        "pairs_per_repeat": args.pairs_per_repeat,
        "overlap_repeats": args.overlap_repeats,
        "requested_device": args.device,
        "resolved_device": device,
        "device_notes": device_notes,
    }
    (logs_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n")

    runtime_s = time.perf_counter() - t0
    write_summary(
        logs_dir / "transferability_summary.md",
        graphs=graphs,
        factor_summary=factor_summary,
        fingerprint_df=fingerprint_df,
        runtime_s=runtime_s,
        device=device,
        notes=device_notes,
    )
    print(f"[transferability] done in {runtime_s:.2f}s -> {out_dir}")


if __name__ == "__main__":
    main()
