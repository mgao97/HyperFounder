from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v2.scripts.run_transferability_analysis import preprocess_graph
from v2.scripts.run_uploaded_raw_benchmark_analysis import load_raw_benchmark, motif_counts_for_graphs


DOMAIN_ORDER = ["coauthorship", "contact", "email", "qa", "threads", "tags"]
DEFAULT_DATASETS = [
    "coauth-DBLP",
    "coauth-geology",
    "coauth-history",
    "contact-high-school",
    "contact-primary-school",
    "email-Enron-full",
    "email-Eu-full",
    "tags-ask-ubuntu",
    "tags-math-sx",
    "threads-ask-ubuntu",
    "threads-math-sx",
]


@dataclass
class FeaturePack:
    size_hist: np.ndarray
    overlap_hist: np.ndarray
    node_deg_hist: np.ndarray
    edge_deg_hist: np.ndarray
    community_hist: np.ndarray
    community_scalars: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def broad_domain(name: str, split_threads: bool = False) -> str:
    n = name.lower()
    if n.startswith("coauth"):
        return "coauthorship"
    if n.startswith("contact"):
        return "contact"
    if n.startswith("email"):
        return "email"
    if n.startswith("tags-"):
        return "tags" if split_threads else "qa"
    if n.startswith("threads-"):
        return "threads" if split_threads else "qa"
    return "unknown"


def order_graphs(graphs: Sequence) -> List:
    rank = {name: i for i, name in enumerate(DOMAIN_ORDER)}
    return sorted(graphs, key=lambda g: (rank.get(g.domain, 999), g.domain, g.name))


def ordered_pairs(graphs: Sequence) -> Iterable[Tuple[object, object]]:
    for g1 in graphs:
        for g2 in graphs:
            if g1.name != g2.name:
                yield g1, g2


def undirected_pairs(graphs: Sequence) -> Iterable[Tuple[object, object]]:
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            yield graphs[i], graphs[j]


def relation(g1, g2) -> str:
    return "within" if g1.domain == g2.domain else "cross"


def pad_pair(a: Sequence[float], b: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    n = max(len(aa), len(bb))
    return np.pad(aa, (0, n - len(aa))), np.pad(bb, (0, n - len(bb)))


def normalize_prob(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    total = float(arr.sum())
    return arr / total if total > 0 else np.zeros_like(arr, dtype=np.float64)


def js_distance(a: Sequence[float], b: Sequence[float]) -> float:
    aa, bb = pad_pair(a, b)
    aa = normalize_prob(aa)
    bb = normalize_prob(bb)
    if aa.sum() <= 0 or bb.sum() <= 0:
        return 1.0
    m = 0.5 * (aa + bb)
    js = 0.5 * stats.entropy(aa, m, base=2) + 0.5 * stats.entropy(bb, m, base=2)
    return float(min(1.0, math.sqrt(max(0.0, float(js)))))


def js_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    return float(max(0.0, 1.0 - js_distance(a, b)))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    aa, bb = pad_pair(a, b)
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(aa, bb) / (na * nb))


def hist(values: Sequence[float], bins: np.ndarray) -> np.ndarray:
    out, _ = np.histogram(np.asarray(values, dtype=np.float64), bins=bins)
    return out.astype(np.float64)


def size_hist(graph) -> np.ndarray:
    sizes = np.asarray([len(e) for e in graph.hyperedges], dtype=np.int64)
    if sizes.size == 0:
        return np.zeros(1, dtype=np.float64)
    bins = np.arange(1, int(sizes.max()) + 2) - 0.5
    return hist(sizes, bins)


def degree_hists(graph) -> Tuple[np.ndarray, np.ndarray]:
    node_deg = np.zeros(graph.num_nodes, dtype=np.int64)
    edge_deg = np.asarray([len(e) for e in graph.hyperedges], dtype=np.int64)
    for edge in graph.hyperedges:
        node_deg[list(edge)] += 1
    n_bins = np.arange(0, int(node_deg.max()) + 2) - 0.5 if node_deg.size else np.array([-0.5, 0.5])
    e_bins = np.arange(1, int(edge_deg.max()) + 2) - 0.5 if edge_deg.size else np.array([0.5, 1.5])
    return hist(node_deg, n_bins), hist(edge_deg, e_bins)


def overlap_hist(graph, sample_pairs: int, seed: int, bins: int = 16) -> np.ndarray:
    m = len(graph.hyperedges)
    if m < 2:
        return np.zeros(bins, dtype=np.float64)
    rng = np.random.default_rng(seed)
    edge_sets = [set(e) for e in graph.hyperedges]
    total = m * (m - 1) // 2
    if total <= sample_pairs:
        pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    else:
        pairs = []
        seen = set()
        while len(pairs) < sample_pairs:
            i = int(rng.integers(0, m))
            j = int(rng.integers(0, m))
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            pairs.append((a, b))
    vals = []
    for i, j in pairs:
        inter = len(edge_sets[i] & edge_sets[j])
        if inter <= 0:
            continue
        union = len(edge_sets[i] | edge_sets[j])
        vals.append(inter / union if union else 0.0)
    if not vals:
        out = np.zeros(bins, dtype=np.float64)
        out[0] = 1.0
        return out
    edges = np.linspace(0.0, 1.0, bins + 1)
    return hist(vals, edges)


def build_bipartite_graph(graph) -> nx.Graph:
    G = nx.Graph()
    offset = graph.num_nodes
    G.add_nodes_from(range(graph.num_nodes), bipartite=0)
    G.add_nodes_from(range(offset, offset + len(graph.hyperedges)), bipartite=1)
    for e_idx, edge in enumerate(graph.hyperedges):
        eid = offset + e_idx
        for node in edge:
            G.add_edge(int(node), eid)
    return G


def community_signature(graph, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    G = build_bipartite_graph(graph)
    if G.number_of_nodes() == 0:
        return np.zeros(1, dtype=np.float64), np.zeros(4, dtype=np.float64)
    communities = list(nx.algorithms.community.asyn_lpa_communities(G, seed=seed))
    if not communities:
        communities = [set(G.nodes())]
    sizes = np.asarray([len(c) for c in communities], dtype=np.float64)
    bins = np.arange(1, int(sizes.max()) + 2) - 0.5
    probs = normalize_prob(sizes)
    entropy = float(stats.entropy(probs)) if probs.size else 0.0
    entropy_norm = float(entropy / math.log(len(sizes))) if len(sizes) > 1 else 0.0
    modularity = float(nx.algorithms.community.modularity(G, communities))
    scalars = np.asarray(
        [
            len(communities) / max(G.number_of_nodes(), 1),
            sizes.max() / max(G.number_of_nodes(), 1),
            entropy_norm,
            modularity,
        ],
        dtype=np.float64,
    )
    return hist(sizes, bins), scalars


def extract_features(graph, sample_pairs: int, seed: int) -> FeaturePack:
    node_deg_hist, edge_deg_hist = degree_hists(graph)
    community_hist, community_scalars = community_signature(graph, seed=seed)
    return FeaturePack(
        size_hist=size_hist(graph),
        overlap_hist=overlap_hist(graph, sample_pairs=sample_pairs, seed=seed),
        node_deg_hist=node_deg_hist,
        edge_deg_hist=edge_deg_hist,
        community_hist=community_hist,
        community_scalars=community_scalars,
    )


def pairwise_similarity(factor: str, a: FeaturePack, b: FeaturePack, motif_a: np.ndarray | None = None, motif_b: np.ndarray | None = None) -> float:
    if factor == "size":
        return js_similarity(a.size_hist, b.size_hist)
    if factor == "overlap":
        return js_similarity(a.overlap_hist, b.overlap_hist)
    if factor == "incidence":
        return 0.5 * (js_similarity(a.node_deg_hist, b.node_deg_hist) + js_similarity(a.edge_deg_hist, b.edge_deg_hist))
    if factor == "community":
        return 0.5 * (js_similarity(a.community_hist, b.community_hist) + cosine_similarity(a.community_scalars, b.community_scalars))
    if factor == "motif":
        assert motif_a is not None and motif_b is not None
        return cosine_similarity(motif_a, motif_b)
    raise KeyError(factor)


def size_count_map(graph) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for edge in graph.hyperedges:
        k = len(edge)
        out[k] = out.get(k, 0) + 1
    return out


def global_min_size_profile(graphs: Sequence) -> Dict[int, int]:
    all_sizes = sorted({len(edge) for g in graphs for edge in g.hyperedges})
    profile: Dict[int, int] = {}
    for size in all_sizes:
        counts = [size_count_map(g).get(size, 0) for g in graphs]
        minimum = min(counts)
        if minimum > 0:
            profile[size] = int(minimum)
    return profile


def resample_to_size_profile(graph, target_profile: Dict[int, int], seed: int):
    rng = np.random.default_rng(seed)
    buckets: Dict[int, List[Tuple[int, ...]]] = {}
    for edge in graph.hyperedges:
        buckets.setdefault(len(edge), []).append(edge)
    sampled = []
    for size, target in target_profile.items():
        edges = buckets.get(size, [])
        use = min(target, len(edges))
        if use <= 0:
            continue
        idx = rng.choice(len(edges), size=use, replace=False)
        sampled.extend(edges[int(i)] for i in np.atleast_1d(idx))
    return preprocess_graph(name=f"{graph.name}__size_matched", domain=graph.domain, num_nodes=graph.num_nodes, hyperedges=sampled)


def degree_preserving_null(graph, seed: int, max_attempts: int = 30):
    rng = np.random.default_rng(seed)
    edge_sizes = [len(edge) for edge in graph.hyperedges]
    node_deg = np.zeros(graph.num_nodes, dtype=np.int64)
    for edge in graph.hyperedges:
        node_deg[list(edge)] += 1
    for _ in range(max_attempts):
        remaining = node_deg.copy()
        seen = set()
        edges = [None] * len(edge_sizes)
        ok = True
        for idx in rng.permutation(len(edge_sizes)):
            size = edge_sizes[int(idx)]
            placed = False
            for _ in range(64):
                candidates = np.flatnonzero(remaining > 0)
                if candidates.size < size:
                    break
                probs = remaining[candidates].astype(np.float64)
                probs = probs / probs.sum()
                chosen = np.sort(rng.choice(candidates, size=size, replace=False, p=probs))
                edge = tuple(int(x) for x in chosen.tolist())
                if edge in seen:
                    continue
                seen.add(edge)
                remaining[chosen] -= 1
                edges[int(idx)] = edge
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok and np.all(remaining == 0):
            return preprocess_graph(name=f"{graph.name}__null", domain=graph.domain, num_nodes=graph.num_nodes, hyperedges=edges)
    raise RuntimeError(f"Failed to build degree-preserving null for {graph.name}")


def mann_whitney_stats(within: Sequence[float], cross: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(within, dtype=np.float64)
    b = np.asarray(cross, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return {"p": float("nan"), "rank_biserial": float("nan"), "cliffs_delta": float("nan")}
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    u = float(res.statistic)
    rank_biserial = float((2.0 * u) / (a.size * b.size) - 1.0)
    gt = 0
    lt = 0
    for x in a:
        gt += int(np.sum(x > b))
        lt += int(np.sum(x < b))
    cliffs = float((gt - lt) / (a.size * b.size))
    return {"p": float(res.pvalue), "rank_biserial": rank_biserial, "cliffs_delta": cliffs}


def block_heatmap(df: pd.DataFrame, domain_map: Dict[str, str], out_path: Path, title: str) -> None:
    order = sorted(df.index.tolist(), key=lambda n: (DOMAIN_ORDER.index(domain_map.get(n, "unknown")) if domain_map.get(n, "unknown") in DOMAIN_ORDER else 999, domain_map.get(n, "unknown"), n))
    mat = df.loc[order, order]
    plt.figure(figsize=(8.5, 7.0))
    ax = sns.heatmap(mat, annot=True, fmt=".2f", cmap="viridis", vmin=0.0, vmax=1.0, square=True, cbar_kws={"shrink": 0.8})
    prev = None
    for idx, name in enumerate(order):
        cur = domain_map.get(name, "unknown")
        if prev is not None and cur != prev:
            ax.axhline(idx, color="white", linewidth=2.0)
            ax.axvline(idx, color="white", linewidth=2.0)
        prev = cur
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_summary_table(pair_df: pd.DataFrame, factors: Sequence[str]) -> pd.DataFrame:
    rows = []
    for factor in factors:
        sub = pair_df[pair_df["factor"] == factor]
        within = sub.loc[sub["relation"] == "within", "similarity"].to_numpy(dtype=np.float64)
        cross = sub.loc[sub["relation"] == "cross", "similarity"].to_numpy(dtype=np.float64)
        stats_row = mann_whitney_stats(within, cross)
        rows.append(
            {
                "factor": factor,
                "S_within_mean": float(within.mean()) if within.size else float("nan"),
                "S_within_std": float(within.std(ddof=1)) if within.size > 1 else 0.0,
                "S_cross_mean": float(cross.mean()) if cross.size else float("nan"),
                "S_cross_std": float(cross.std(ddof=1)) if cross.size > 1 else 0.0,
                "delta_within_minus_cross": float(within.mean() - cross.mean()) if within.size and cross.size else float("nan"),
                "mannwhitney_p": stats_row["p"],
                "rank_biserial": stats_row["rank_biserial"],
                "cliffs_delta": stats_row["cliffs_delta"],
                "n_within_ordered": int(within.size),
                "n_cross_ordered": int(cross.size),
            }
        )
    return pd.DataFrame(rows)


def leave_one_out_summary(pair_df: pd.DataFrame, graphs: Sequence, factors: Sequence[str]) -> pd.DataFrame:
    rows = []
    for dropped in graphs:
        sub = pair_df[(pair_df["dataset_a"] != dropped.name) & (pair_df["dataset_b"] != dropped.name)]
        summary = build_summary_table(sub, factors)
        for _, row in summary.iterrows():
            rows.append({"dropped_dataset": dropped.name, **row.to_dict()})
    return pd.DataFrame(rows)


def write_summary(out_path: Path, dataset_df: pd.DataFrame, summary_df: pd.DataFrame, size_match_df: pd.DataFrame | None, null_df: pd.DataFrame | None, available: Sequence[str], missing: Sequence[str], domain_mode: str) -> None:
    lines = [
        "# Broad-Domain Benchmark Transferability Summary",
        "",
        f"- Domain mode: `{domain_mode}`",
        f"- Available datasets: {', '.join(available)}",
        f"- Missing datasets: {', '.join(missing) if missing else 'none'}",
        "",
        "## Dataset Stats",
        "",
        dataset_df.to_markdown(index=False),
        "",
        "## Factor Summary",
        "",
        summary_df.to_markdown(index=False),
    ]
    if size_match_df is not None and not size_match_df.empty:
        lines.extend(["", "## Size-Matched Motif Summary", "", size_match_df.to_markdown(index=False)])
    if null_df is not None and not null_df.empty:
        lines.extend(["", "## Null Excess Summary", "", null_df.to_markdown(index=False)])
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, default="data/benchmark_raw")
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overlap_pairs", type=int, default=120000)
    parser.add_argument("--num_threads", type=int, default=32)
    parser.add_argument("--mochy_repo", type=str, default="/home/user/GSK/mgao/MoCHy_ref")
    parser.add_argument("--output_dir", type=str, default="outputs_transferability/broad_domain_benchmark_seed7")
    parser.add_argument("--domain_mode", choices=["wide", "split_threads"], default="wide")
    parser.add_argument("--null_replicates", type=int, default=0)
    parser.add_argument("--skip_missing", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    raw_root = ROOT / args.raw_root
    out_dir = ROOT / args.output_dir
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    logs_dir = out_dir / "logs"
    for folder in (tables_dir, figures_dir, logs_dir):
        folder.mkdir(parents=True, exist_ok=True)

    requested = [x.strip() for x in args.datasets.split(",") if x.strip()]
    available = [x for x in requested if (raw_root / x).exists()]
    missing = [x for x in requested if not (raw_root / x).exists()]
    if missing and not args.skip_missing:
        raise FileNotFoundError(f"Missing datasets: {missing}")

    graphs = []
    rows = []
    for name in available:
        raw_g, proc_g = load_raw_benchmark(raw_root / name)
        graph = preprocess_graph(
            name=proc_g.name,
            domain=proc_g.domain,
            num_nodes=proc_g.num_nodes,
            hyperedges=proc_g.hyperedges,
        )
        graphs.append(graph)
        sizes = [len(e) for e in graph.hyperedges]
        rows.append(
            {
                "dataset": graph.name,
                "domain": graph.domain,
                "num_nodes": graph.num_nodes,
                "num_hyperedges": len(graph.hyperedges),
                "avg_hyperedge_size": float(np.mean(sizes)) if sizes else 0.0,
                "max_hyperedge_size": max(sizes) if sizes else 0,
                "raw_label_count": raw_g.n_labels,
            }
        )
    graphs = order_graphs(graphs)
    dataset_df = pd.DataFrame(rows).sort_values(["domain", "dataset"]).reset_index(drop=True)
    dataset_df.to_csv(tables_dir / "dataset_basic_stats.csv", index=False)

    features = {g.name: extract_features(g, sample_pairs=args.overlap_pairs, seed=args.seed + i * 97) for i, g in enumerate(graphs)}
    motif_df, motif_map = motif_counts_for_graphs(graphs, Path(args.mochy_repo), out_dir, num_threads=args.num_threads)
    motif_df.to_csv(out_dir / "motif_counts_and_cp.csv", index=False)

    names = [g.name for g in graphs]
    index = pd.Index(names, name="dataset")
    factors = ["size", "overlap", "incidence", "community", "motif"]
    matrices = {factor: pd.DataFrame(np.eye(len(names)), index=index, columns=names) for factor in factors}
    for g1, g2 in undirected_pairs(graphs):
        for factor in factors:
            sim = pairwise_similarity(factor, features[g1.name], features[g2.name], motif_map[g1.name], motif_map[g2.name])
            matrices[factor].loc[g1.name, g2.name] = sim
            matrices[factor].loc[g2.name, g1.name] = sim
    domain_map = {g.name: g.domain for g in graphs}
    pair_rows = []
    undirected_rows = []
    for factor, mat in matrices.items():
        mat.to_csv(tables_dir / f"{factor}_similarity_matrix.csv")
        block_heatmap(mat, domain_map, figures_dir / f"{factor}_similarity_heatmap.png", f"{factor.capitalize()} Similarity")
        for g1, g2 in ordered_pairs(graphs):
            pair_rows.append(
                {
                    "factor": factor,
                    "dataset_a": g1.name,
                    "dataset_b": g2.name,
                    "domain_a": g1.domain,
                    "domain_b": g2.domain,
                    "relation": relation(g1, g2),
                    "similarity": float(mat.loc[g1.name, g2.name]),
                }
            )
        for g1, g2 in undirected_pairs(graphs):
            undirected_rows.append(
                {
                    "factor": factor,
                    "dataset_a": g1.name,
                    "dataset_b": g2.name,
                    "domain_a": g1.domain,
                    "domain_b": g2.domain,
                    "relation": relation(g1, g2),
                    "similarity": float(mat.loc[g1.name, g2.name]),
                }
            )
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(tables_dir / "pair_level_scores_ordered.csv", index=False)
    pd.DataFrame(undirected_rows).to_csv(tables_dir / "pair_level_scores_undirected.csv", index=False)

    summary_df = build_summary_table(pair_df, factors)
    summary_df.to_csv(tables_dir / "factor_summary.csv", index=False)
    leave_one_out_summary(pair_df, graphs, factors).to_csv(tables_dir / "leave_one_dataset_out_summary.csv", index=False)

    size_match_summary = None
    target_profile = global_min_size_profile(graphs)
    if target_profile:
        matched_graphs = [resample_to_size_profile(g, target_profile, seed=args.seed + i * 1009) for i, g in enumerate(graphs)]
        _, matched_motif = motif_counts_for_graphs(matched_graphs, Path(args.mochy_repo), out_dir / "size_matched_motif", num_threads=args.num_threads)
        rows = []
        for g1, g2 in ordered_pairs(matched_graphs):
            rows.append(
                {
                    "dataset_a": g1.name.replace("__size_matched", ""),
                    "dataset_b": g2.name.replace("__size_matched", ""),
                    "relation": relation(g1, g2),
                    "similarity": cosine_similarity(matched_motif[g1.name], matched_motif[g2.name]),
                }
            )
        sm_pair = pd.DataFrame(rows)
        sm_pair.to_csv(tables_dir / "motif_size_matched_pair_scores.csv", index=False)
        tmp = sm_pair.rename(columns={"similarity": "similarity"})
        size_match_summary = build_summary_table(tmp.assign(factor="motif_size_matched"), ["motif_size_matched"])
        size_match_summary["target_size_profile"] = ";".join(f"{k}:{v}" for k, v in sorted(target_profile.items()))
        size_match_summary.to_csv(tables_dir / "motif_size_matched_summary.csv", index=False)

    null_summary = None
    if args.null_replicates > 0:
        null_rows = []
        cross_pairs = [(g1, g2) for g1, g2 in ordered_pairs(graphs) if relation(g1, g2) == "cross"]
        for rep in range(args.null_replicates):
            null_graphs = {g.name: degree_preserving_null(g, seed=args.seed + rep * 10007 + i * 37) for i, g in enumerate(graphs)}
            null_features = {name: extract_features(g, sample_pairs=args.overlap_pairs, seed=args.seed + rep * 307 + i * 13) for i, (name, g) in enumerate(null_graphs.items())}
            _, null_motif = motif_counts_for_graphs(list(null_graphs.values()), Path(args.mochy_repo), out_dir / f"null_models/rep_{rep:02d}", num_threads=args.num_threads)
            for g1, g2 in cross_pairs:
                for factor in factors:
                    null_rows.append(
                        {
                            "replicate": rep,
                            "factor": factor,
                            "dataset_a": g1.name,
                            "dataset_b": g2.name,
                            "similarity": pairwise_similarity(factor, features[g1.name], null_features[g2.name], motif_map[g1.name], null_motif[f"{g2.name}__null"]),
                        }
                    )
        null_pair_df = pd.DataFrame(null_rows)
        null_pair_df.to_csv(tables_dir / "null_pair_level_scores.csv", index=False)
        rows = []
        for factor in factors:
            cross_mean = float(pair_df.loc[(pair_df["factor"] == factor) & (pair_df["relation"] == "cross"), "similarity"].mean())
            null_mean = float(null_pair_df.loc[null_pair_df["factor"] == factor, "similarity"].mean())
            rows.append({"factor": factor, "S_cross_mean": cross_mean, "S_null_mean": null_mean, "S_cross_minus_S_null": cross_mean - null_mean, "null_replicates": args.null_replicates})
        null_summary = pd.DataFrame(rows)
        null_summary.to_csv(tables_dir / "null_excess_summary.csv", index=False)

    write_summary(logs_dir / "broad_domain_summary.md", dataset_df, summary_df, size_match_summary, null_summary, available, missing, args.domain_mode)
    print(f"[broad-domain-analysis] done -> {out_dir}")


if __name__ == "__main__":
    main()
