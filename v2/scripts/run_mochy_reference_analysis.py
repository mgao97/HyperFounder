from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v2.scripts.run_transferability_analysis import preprocess_graph
from v2.utils.dhg_datasets import load_dhg_sample
from v2.utils.dataset_registry import get_dataset_spec


BENCHMARK_ROWS = [
    ("coauth-DBLP", "coauthorship", 1924991, 2400000, 125000000, 26300000000, "https://www.cs.cornell.edu/~arb/data/coauth-DBLP/index.html"),
    ("coauth-geology", "coauthorship", 1200000, 1200000, 37600000, 6000000000, "https://www.cs.cornell.edu/~arb/data/coauth-MAG-Geology/index.html"),
    ("coauth-history", "coauthorship", 1000000, 895000, 1700000, 83200000, "https://www.cs.cornell.edu/~arb/data/coauth-MAG-History/index.html"),
    ("contact-primary", "contact", 242, 12700, 2200000, 617000000, "https://www.cs.cornell.edu/~arb/data/contact-primary-school/index.html"),
    ("contact-high", "contact", 327, 7800, 593000, 69700000, "https://www.cs.cornell.edu/~arb/data/contact-high-school/index.html"),
    ("email-Enron", "email", 143, 1500, 87800, 9600000, "https://www.cs.cornell.edu/~arb/data/email-Enron/index.html"),
    ("email-EU", "email", 998, 25000, 8300000, 7000000000, "https://www.cs.cornell.edu/~arb/data/email-EU/index.html"),
    ("tags-ask-ubuntu", "tags", 3000, 147000, 564000000, 4300000000000, "https://www.cs.cornell.edu/~arb/data/tags-ask-ubuntu/index.html"),
    ("tags-math-sx", "tags", 1600, 170000, 913000000, 9200000000000, "https://www.cs.cornell.edu/~arb/data/tags-math-sx/index.html"),
    ("threads-ask-ubuntu", "threads", 125000, 166000, 21600000, 11400000000, "https://www.cs.cornell.edu/~arb/data/threads-ask-ubuntu/index.html"),
    ("threads-math-sx", "threads", 176000, 595000, 647000000, 2200000000000, "https://www.cs.cornell.edu/~arb/data/threads-math-sx/index.html"),
]


ID_TO_INDEX = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 21, 23, 22, 24, 23, 25, 24, 26,
    0, 0, 0, 0, 0, 0, 0, 0, 21, 22, 23, 24, 23, 24, 25, 26,
    21, 23, 23, 25, 22, 24, 24, 26, 27, 28, 28, 29, 28, 29, 29, 30,
    1, 2, 2, 3, 2, 3, 3, 4, 5, 6, 6, 8, 7, 9, 9, 10,
    5, 7, 6, 9, 6, 9, 8, 10, 11, 13, 12, 14, 13, 15, 14, 16,
    5, 6, 7, 9, 6, 8, 9, 10, 11, 12, 13, 14, 13, 14, 15, 16,
    11, 13, 13, 15, 12, 14, 14, 16, 17, 18, 18, 19, 18, 19, 19, 20,
]

MOTIF_SKIP = {0, 1, 4, 6}
EXPECTED_SAMPLE_COUNTS = np.array(
    [2251, 743473, 370, 169, 91020, 73208, 858, 560, 2007, 1726, 1563, 1978, 11, 25, 36, 39, 84, 78, 18621, 16034, 471169, 825742, 306, 1567, 2815, 1829],
    dtype=np.int64,
)


@dataclass
class PreparedGraph:
    name: str
    domain: str
    num_nodes: int
    hyperedges: List[Tuple[int, ...]]


def read_mochy_file(path: Path) -> List[Tuple[int, ...]]:
    hyperedges: List[Tuple[int, ...]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            edge = tuple(sorted({int(x) for x in line.split(",") if x != ""}))
            if len(edge) >= 2:
                hyperedges.append(edge)
    return hyperedges


def motif_index(size_a: int, size_b: int, size_c: int, c_ab: int, c_bc: int, c_ca: int, g_abc: int) -> int:
    a = size_a - (c_ab + c_ca) + g_abc
    b = size_b - (c_bc + c_ab) + g_abc
    c = size_c - (c_ca + c_bc) + g_abc
    d = c_ab - g_abc
    e = c_bc - g_abc
    f = c_ca - g_abc
    g = g_abc
    motif_id = (a > 0) + ((b > 0) << 1) + ((c > 0) << 2) + ((d > 0) << 3) + ((e > 0) << 4) + ((f > 0) << 5) + ((g > 0) << 6)
    return ID_TO_INDEX[motif_id] - 1


def build_incidence(edges: Sequence[Tuple[int, ...]]) -> Tuple[List[List[int]], List[set[int]]]:
    max_node = max(max(edge) for edge in edges) if edges else -1
    node2hyperedge: List[List[int]] = [[] for _ in range(max_node + 1)]
    hyperedge_sets: List[set[int]] = []
    for e_idx, edge in enumerate(edges):
        edge_set = set(edge)
        hyperedge_sets.append(edge_set)
        for node in edge_set:
            node2hyperedge[node].append(e_idx)
    return node2hyperedge, hyperedge_sets


def build_adjacency(edges: Sequence[Tuple[int, ...]], node2hyperedge: Sequence[Sequence[int]]) -> Tuple[List[List[Tuple[int, int]]], List[Dict[int, int]]]:
    inter_maps: List[Dict[int, int]] = [defaultdict(int) for _ in range(len(edges))]
    for inc in node2hyperedge:
        if len(inc) < 2:
            continue
        for i, a in enumerate(inc):
            for b in inc[i + 1:]:
                inter_maps[a][b] += 1
                inter_maps[b][a] += 1
    adj = [sorted(m.items()) for m in inter_maps]
    return adj, inter_maps


def python_exact_motif_counts(edges: Sequence[Tuple[int, ...]]) -> np.ndarray:
    if not edges:
        return np.zeros(26, dtype=np.int64)

    node2hyperedge, edge_sets = build_incidence(edges)
    adj, inter_maps = build_adjacency(edges, node2hyperedge)
    counts = np.zeros(30, dtype=np.int64)

    for a in range(len(edges)):
        set_a = edge_sets[a]
        size_a = len(set_a)
        neigh_a = adj[a]
        for i, (b, c_ab) in enumerate(neigh_a):
            set_b = edge_sets[b]
            size_b = len(set_b)
            common_ab = set_a & set_b
            map_b = inter_maps[b]
            for c, c_ca in neigh_a[i + 1:]:
                set_c = edge_sets[c]
                size_c = len(set_c)
                c_bc = map_b.get(c, 0)
                if c_bc:
                    if a < b:
                        g_abc = sum(1 for node in common_ab if node in set_c)
                        idx = motif_index(size_a, size_b, size_c, c_ab, c_bc, c_ca, g_abc)
                        counts[idx] += 1
                else:
                    idx = motif_index(size_a, size_b, size_c, c_ab, 0, c_ca, 0)
                    counts[idx] += 1

    out = []
    for i in range(30):
        if i in MOTIF_SKIP:
            continue
        out.append(int(counts[i]))
    return np.asarray(out, dtype=np.int64)


def ensure_cpp_binary(repo_dir: Path) -> Path:
    binary = repo_dir / "main_exact_bin"
    if binary.exists():
        return binary
    cmd = ["g++", "-O3", "-std=c++17", str(repo_dir / "main_exact.cpp"), "-o", str(binary)]
    subprocess.run(cmd, check=True, cwd=repo_dir)
    return binary


def run_cpp_exact(binary: Path, input_path: Path) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="mochy_cpp_") as tmpdir:
        tmp = Path(tmpdir)
        target = tmp / "dblp_graph.txt"
        target.write_text(input_path.read_text())
        proc = subprocess.run([str(binary)], cwd=tmp, check=True, capture_output=True, text=True)
    counts = []
    for line in proc.stdout.splitlines():
        m = re.match(r"motif\s+(\d+):\s+(\d+)", line.strip())
        if m:
            counts.append(int(m.group(2)))
    if len(counts) != 26:
        raise RuntimeError(f"C++ output parse failed, got {len(counts)} motif counts")
    return np.asarray(counts, dtype=np.int64)


def export_edges(path: Path, edges: Sequence[Tuple[int, ...]]) -> None:
    with open(path, "w") as f:
        for edge in edges:
            f.write(",".join(map(str, edge)) + "\n")


def load_local_graphs(dataset_names: Sequence[str], target_dim: int, cache_dir: str, seed: int) -> List[PreparedGraph]:
    graphs: List[PreparedGraph] = []
    for name in dataset_names:
        spec = get_dataset_spec(name)
        raw = load_dhg_sample(name, target_dim=target_dim, seed=seed, data_root=cache_dir, require_node_splits=False)
        pre = preprocess_graph(name=name, domain=spec.domain, num_nodes=raw.num_nodes, hyperedges=raw.hyperedges)
        graphs.append(PreparedGraph(name=pre.name, domain=pre.domain, num_nodes=pre.num_nodes, hyperedges=pre.hyperedges))
    return graphs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_similarity(cp_map: Dict[str, np.ndarray], metric: str) -> pd.DataFrame:
    names = list(cp_map.keys())
    mat = np.eye(len(names), dtype=np.float64)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = cp_map[names[i]]
            b = cp_map[names[j]]
            if metric == "cosine":
                val = cosine_similarity(a, b)
            elif metric == "pearson":
                if np.allclose(a, a[0]) or np.allclose(b, b[0]):
                    val = 0.0
                else:
                    val = float(np.corrcoef(a, b)[0, 1])
            elif metric == "euclidean_similarity":
                val = float(math.exp(-np.linalg.norm(a - b)))
            else:
                raise ValueError(metric)
            mat[i, j] = val
            mat[j, i] = val
    return pd.DataFrame(mat, index=names, columns=names)


def save_heatmap(df: pd.DataFrame, out_path: Path, title: str, vmin: float | None = None, vmax: float | None = None) -> None:
    plt.figure(figsize=(7.5, 6.5))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", square=True, vmin=vmin, vmax=vmax, cbar_kws={"shrink": 0.8})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def factor_summary_from_matrix(graphs: Sequence[PreparedGraph], matrix: pd.DataFrame) -> Dict[str, float]:
    within = []
    cross = []
    for g1, g2 in combinations(graphs, 2):
        val = float(matrix.loc[g1.name, g2.name])
        if g1.domain == g2.domain:
            within.append(val)
        else:
            cross.append(val)
    mu_within = float(np.mean(within)) if within else float("nan")
    mu_cross = float(np.mean(cross)) if cross else float("nan")
    pvalue = float(stats.mannwhitneyu(within, cross, alternative="two-sided").pvalue) if within and cross else float("nan")
    return {
        "mu_within": mu_within,
        "mu_cross": mu_cross,
        "delta_within_minus_cross": mu_within - mu_cross if within and cross else float("nan"),
        "mannwhitney_p": pvalue,
        "n_within_pairs": len(within),
        "n_cross_pairs": len(cross),
    }


def benchmark_consistency(local_graphs: Sequence[PreparedGraph]) -> pd.DataFrame:
    instruction_names = {
        "coauth-DBLP", "coauth-geology", "coauth-history",
        "contact-high", "contact-primary", "email-Enron", "email-EU",
        "tags-ask-ubuntu", "tags-math-sx", "threads-ask-ubuntu", "threads-math-sx",
    }
    local_rows = {g.name: g for g in local_graphs}
    rows = []
    for dataset, domain, nodes, edges, hyperwedges, motifs, source in BENCHMARK_ROWS:
        if dataset in local_rows:
            status = "exact_name_match"
            local_name = dataset
            local_nodes = local_rows[dataset].num_nodes
            local_edges = len(local_rows[dataset].hyperedges)
        else:
            status = "missing_locally"
            local_name = ""
            local_nodes = np.nan
            local_edges = np.nan
        rows.append(
            {
                "benchmark_dataset": dataset,
                "domain": domain,
                "benchmark_nodes": nodes,
                "benchmark_hyperedges": edges,
                "benchmark_hyperwedges": hyperwedges,
                "benchmark_hmotifs": motifs,
                "in_instruction_list": dataset in instruction_names,
                "local_name_match": local_name,
                "local_nodes": local_nodes,
                "local_hyperedges": local_edges,
                "status": status,
                "source": source,
            }
        )
    for g in local_graphs:
        if g.name not in {row["benchmark_dataset"] for row in rows}:
            rows.append(
                {
                    "benchmark_dataset": "",
                    "domain": g.domain,
                    "benchmark_nodes": np.nan,
                    "benchmark_hyperedges": np.nan,
                    "benchmark_hyperwedges": np.nan,
                    "benchmark_hmotifs": np.nan,
                    "in_instruction_list": False,
                    "local_name_match": g.name,
                    "local_nodes": g.num_nodes,
                    "local_hyperedges": len(g.hyperedges),
                    "status": "local_only_non_mochy_benchmark",
                    "source": "",
                }
            )
    return pd.DataFrame(rows)


def validate_python_vs_cpp(repo_dir: Path, out_dir: Path) -> pd.DataFrame:
    sample_path = repo_dir / "dblp_graph.txt"
    sample_edges = read_mochy_file(sample_path)
    t0 = time.perf_counter()
    py_counts = python_exact_motif_counts(sample_edges)
    py_time = time.perf_counter() - t0
    binary = ensure_cpp_binary(repo_dir)
    t1 = time.perf_counter()
    cpp_counts = run_cpp_exact(binary, sample_path)
    cpp_time = time.perf_counter() - t1
    ok_cpp = np.array_equal(py_counts, cpp_counts)
    ok_expected = np.array_equal(py_counts, EXPECTED_SAMPLE_COUNTS)
    df = pd.DataFrame(
        {
            "motif_index": np.arange(1, 27),
            "python_count": py_counts,
            "cpp_count": cpp_counts,
            "expected_readme_count": EXPECTED_SAMPLE_COUNTS,
            "python_eq_cpp": py_counts == cpp_counts,
            "python_eq_expected": py_counts == EXPECTED_SAMPLE_COUNTS,
        }
    )
    df.to_csv(out_dir / "sample_validation_counts.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "python_eq_cpp": bool(ok_cpp),
                "python_eq_expected": bool(ok_expected),
                "python_runtime_s": py_time,
                "cpp_runtime_s": cpp_time,
                "sample_hyperedges": len(sample_edges),
                "sample_nodes": max(max(e) for e in sample_edges) + 1 if sample_edges else 0,
            }
        ]
    )
    summary.to_csv(out_dir / "sample_validation_summary.csv", index=False)
    return summary


def compute_local_motif_results(
    graphs: Sequence[PreparedGraph],
    repo_dir: Path,
    out_dir: Path,
    engine: str,
    python_edge_threshold: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], pd.DataFrame]:
    binary = ensure_cpp_binary(repo_dir)
    result_rows = []
    cp_map: Dict[str, np.ndarray] = {}

    inputs_dir = out_dir / "mochy_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    for graph in graphs:
        input_path = inputs_dir / f"{graph.name}.txt"
        export_edges(input_path, graph.hyperedges)
        use_engine = engine
        if engine == "auto":
            use_engine = "python" if len(graph.hyperedges) <= python_edge_threshold else "cpp_oracle"
        t0 = time.perf_counter()
        if use_engine == "python":
            counts = python_exact_motif_counts(graph.hyperedges)
        elif use_engine == "cpp_oracle":
            counts = run_cpp_exact(binary, input_path)
        else:
            raise ValueError(use_engine)
        dt = time.perf_counter() - t0
        cp = counts.astype(np.float64)
        cp = cp / cp.sum() if cp.sum() > 0 else cp
        cp_map[graph.name] = cp
        row = {
            "dataset": graph.name,
            "domain": graph.domain,
            "num_nodes": graph.num_nodes,
            "num_hyperedges": len(graph.hyperedges),
            "engine": use_engine,
            "runtime_s": dt,
            "total_motif_instances": int(counts.sum()),
        }
        for idx, value in enumerate(counts, start=1):
            row[f"motif_{idx}"] = int(value)
            row[f"cp_{idx}"] = float(cp[idx - 1])
        result_rows.append(row)

    counts_df = pd.DataFrame(result_rows)
    counts_df.to_csv(out_dir / "motif_counts_and_cp.csv", index=False)

    cp_df = pd.DataFrame(
        [{"dataset": name, **{f"cp_{i+1}": float(v) for i, v in enumerate(cp)}} for name, cp in cp_map.items()]
    )
    cp_df.to_csv(out_dir / "motif_characteristic_profiles.csv", index=False)
    return counts_df, cp_map, cp_df


def domain_classification(cp_df: pd.DataFrame, graphs: Sequence[PreparedGraph]) -> pd.DataFrame:
    domain_map = {g.name: g.domain for g in graphs}
    X = cp_df[[c for c in cp_df.columns if c.startswith("cp_")]].to_numpy(dtype=np.float64)
    y_labels = [domain_map[name] for name in cp_df["dataset"]]
    label_to_id = {label: idx for idx, label in enumerate(sorted(set(y_labels)))}
    y = np.array([label_to_id[label] for label in y_labels], dtype=np.int64)
    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    clf = LogisticRegression(max_iter=1000, multi_class="auto")
    for tr, te in loo.split(X):
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])
    acc = accuracy_score(y, preds)
    return pd.DataFrame(
        [
            {
                "n_datasets": len(y),
                "n_domains": len(label_to_id),
                "leave_one_out_accuracy": acc,
            }
        ]
    )


def write_summary(
    out_path: Path,
    validation_summary: pd.DataFrame,
    consistency_df: pd.DataFrame,
    motif_counts_df: pd.DataFrame,
    cosine_summary: Dict[str, float],
    pearson_summary: Dict[str, float],
    clf_df: pd.DataFrame,
) -> None:
    lines = [
        "# MoCHy Reference Alignment Summary",
        "",
        "## 1. Python vs C++ Validation",
        "",
        validation_summary.to_markdown(index=False),
        "",
        "## 2. Benchmark Consistency",
        "",
        consistency_df.to_markdown(index=False),
        "",
        "## 3. Local Motif Counting",
        "",
        motif_counts_df[["dataset", "domain", "num_nodes", "num_hyperedges", "engine", "runtime_s", "total_motif_instances"]].to_markdown(index=False),
        "",
        "## 4. Motif Similarity Summary",
        "",
        pd.DataFrame(
            [
                {"metric": "cosine", **cosine_summary},
                {"metric": "pearson", **pearson_summary},
            ]
        ).to_markdown(index=False),
        "",
        "## 5. Domain Predictability from CP",
        "",
        clf_df.to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- `python` 引擎是对 MoCHy-E exact counting 逻辑的直接 Python 复现。",
        "- `cpp_oracle` 由 Python 管线导出输入并调用官方 C++ exact 实现，主要用于今天先把较大数据集结果稳妥跑出来。",
        "- 当前本地缓存数据与论文 11 个 benchmark 名单并不一致，不能把两者混称为同一 benchmark。",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", type=str, default="/home/user/GSK/mgao/MoCHy_ref")
    parser.add_argument("--datasets", type=str, default="cora_cc,citeseer_cc,coauthorship_cora,coauthorship_dblp,cooking_200,gowalla")
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--target_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--engine", type=str, default="auto", choices=["auto", "python", "cpp_oracle"])
    parser.add_argument("--python_edge_threshold", type=int, default=6000)
    parser.add_argument("--output_dir", type=str, default="outputs_transferability/mochy_reference_seed7")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = Path(args.repo_dir)
    validation_summary = validate_python_vs_cpp(repo_dir, out_dir)
    local_graphs = load_local_graphs([x.strip() for x in args.datasets.split(",") if x.strip()], args.target_dim, args.cache_dir, args.seed)
    consistency_df = benchmark_consistency(local_graphs)
    consistency_df.to_csv(out_dir / "benchmark_consistency.csv", index=False)

    motif_counts_df, cp_map, cp_df = compute_local_motif_results(local_graphs, repo_dir, out_dir, args.engine, args.python_edge_threshold)
    clf_df = domain_classification(cp_df, local_graphs)
    clf_df.to_csv(out_dir / "motif_domain_classification.csv", index=False)

    cosine_df = compute_similarity(cp_map, "cosine")
    pearson_df = compute_similarity(cp_map, "pearson")
    euclid_df = compute_similarity(cp_map, "euclidean_similarity")
    cosine_df.to_csv(out_dir / "motif_similarity_cosine.csv")
    pearson_df.to_csv(out_dir / "motif_similarity_pearson.csv")
    euclid_df.to_csv(out_dir / "motif_similarity_euclidean_similarity.csv")

    save_heatmap(cosine_df, out_dir / "motif_similarity_cosine.png", "Motif CP Cosine Similarity", vmin=0.0, vmax=1.0)
    save_heatmap(pearson_df, out_dir / "motif_similarity_pearson.png", "Motif CP Pearson Similarity", vmin=-1.0, vmax=1.0)

    cosine_summary = factor_summary_from_matrix(local_graphs, cosine_df)
    pearson_summary = factor_summary_from_matrix(local_graphs, pearson_df)
    pd.DataFrame([{"metric": "cosine", **cosine_summary}, {"metric": "pearson", **pearson_summary}]).to_csv(
        out_dir / "motif_similarity_summary.csv", index=False
    )

    write_summary(
        out_dir / "mochy_reference_summary.md",
        validation_summary=validation_summary,
        consistency_df=consistency_df,
        motif_counts_df=motif_counts_df,
        cosine_summary=cosine_summary,
        pearson_summary=pearson_summary,
        clf_df=clf_df,
    )
    print(f"[mochy-reference] done -> {out_dir}")


if __name__ == "__main__":
    main()
