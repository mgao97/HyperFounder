from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v2.scripts.run_mochy_reference_analysis import (
    compute_similarity,
    ensure_cpp_binary,
    export_edges,
    python_exact_motif_counts,
    run_cpp_exact,
)
from v2.scripts.run_transferability_analysis import (
    community_features,
    compute_similarity_matrices,
    incidence_features,
    preprocess_graph,
    save_heatmap,
    save_size_distribution,
    size_features,
    sample_overlap_features,
)


@dataclass
class RawBenchmarkGraph:
    name: str
    domain: str
    num_nodes: int
    hyperedges: List[tuple[int, ...]]
    n_labels: int
    n_label_names: int


def infer_domain(name: str) -> str:
    n = name.lower()
    if "contact" in n:
        return "contact"
    if "email" in n:
        return "email"
    if "coauth" in n:
        return "coauthorship"
    if "thread" in n:
        return "threads"
    if "tag" in n:
        return "tags"
    if "senate" in n or "bill" in n:
        return "political"
    return "unknown"


def read_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in f]


def read_int_lines(path: Path) -> List[int]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [int(line.strip()) for line in f if line.strip()]


def find_single(pattern: str, folder: Path) -> Path | None:
    matches = sorted(folder.glob(pattern))
    return matches[0] if matches else None


def load_raw_benchmark(folder: Path):
    hyperedge_file = find_single("hyperedges-*.txt", folder)
    labels_file = find_single("node-labels-*.txt", folder)
    label_names_file = find_single("label-names-*.txt", folder)

    edges = []
    if hyperedge_file is not None:
        with open(hyperedge_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                edge = tuple(sorted({int(x) for x in line.split(",") if x != ""}))
                if len(edge) >= 2:
                    edges.append(edge)
        if labels_file is not None:
            n_labels = len(read_lines(labels_file))
        else:
            n_labels = max((max(e) for e in edges), default=-1) + 1
        n_label_names = len(read_lines(label_names_file)) if label_names_file is not None else 0
    else:
        nverts_file = find_single("*-nverts.txt", folder)
        simplices_file = find_single("*-simplices.txt", folder)
        if nverts_file is None or simplices_file is None:
            raise FileNotFoundError(f"No supported hypergraph files found in {folder}")
        nverts = read_int_lines(nverts_file)
        simplex_nodes = read_int_lines(simplices_file)
        cursor = 0
        for size in nverts:
            edge_nodes = simplex_nodes[cursor:cursor + size]
            cursor += size
            edge = tuple(sorted(set(int(x) - 1 for x in edge_nodes)))
            if len(edge) >= 2:
                edges.append(edge)
        if labels_file is not None:
            first = read_lines(labels_file)[:5]
            if first and " " in first[0]:
                ids = []
                for line in read_lines(labels_file):
                    token = line.split()[0]
                    ids.append(int(token))
                n_labels = max(ids) if ids else max((max(e) for e in edges), default=-1) + 1
            else:
                n_labels = len(read_lines(labels_file))
        else:
            n_labels = max((max(e) for e in edges), default=-1) + 1
        n_label_names = 0

    try:
        from v2.utils.dataset_registry import get_dataset_spec
        _domain = get_dataset_spec(folder.name).domain
    except Exception:
        _domain = infer_domain(folder.name)
    pre = preprocess_graph(
        name=folder.name,
        domain=_domain,
        num_nodes=n_labels,
        hyperedges=edges,
    )
    return (
        RawBenchmarkGraph(
            name=pre.name,
            domain=pre.domain,
            num_nodes=pre.num_nodes,
            hyperedges=pre.hyperedges,
            n_labels=n_labels,
            n_label_names=n_label_names,
        ),
        pre,
    )


def ensure_cpp_parallel_binary(repo_dir: Path) -> Path:
    binary = repo_dir / "main_exact_par_bin"
    if binary.exists():
        return binary
    cmd = ["g++", "-O3", "-std=c++17", "-lgomp", "-fopenmp", str(repo_dir / "main_exact_par.cpp"), "-o", str(binary)]
    subprocess.run(cmd, check=True, cwd=repo_dir)
    return binary


def run_cpp_exact_parallel(binary: Path, input_path: Path, num_threads: int) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="mochy_par_") as tmpdir:
        tmp = Path(tmpdir)
        target = tmp / "dblp_graph.txt"
        shutil.copy2(input_path, target)
        env = dict(**os.environ, OMP_NUM_THREADS=str(num_threads))
        proc = subprocess.run([str(binary), str(num_threads)], cwd=tmp, check=True, capture_output=True, text=True, env=env)
    counts = []
    for line in proc.stdout.splitlines():
        if line.startswith("motif "):
            counts.append(int(line.split(":")[1].strip()))
    if len(counts) != 26:
        raise RuntimeError(f"Parallel C++ output parse failed, got {len(counts)} motif counts")
    return np.asarray(counts, dtype=np.int64)


def motif_counts_for_graphs(graphs, repo_dir: Path, out_dir: Path, threshold: int = 6000, num_threads: int = 32):
    inputs_dir = out_dir / "mochy_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    binary_parallel = ensure_cpp_parallel_binary(repo_dir)
    rows = []
    cp_map: Dict[str, np.ndarray] = {}
    for graph in graphs:
        input_path = inputs_dir / f"{graph.name}.txt"
        export_edges(input_path, graph.hyperedges)
        engine = "python" if len(graph.hyperedges) <= threshold else "cpp_oracle"
        t0 = time.perf_counter()
        if engine == "python":
            counts = python_exact_motif_counts(graph.hyperedges)
        else:
            counts = run_cpp_exact_parallel(binary_parallel, input_path, num_threads=num_threads)
        dt = time.perf_counter() - t0
        cp = counts.astype(np.float64)
        cp = cp / cp.sum() if cp.sum() > 0 else cp
        cp_map[graph.name] = cp
        row = {
            "dataset": graph.name,
            "domain": graph.domain,
            "num_nodes": graph.num_nodes,
            "num_hyperedges": len(graph.hyperedges),
            "engine": engine,
            "runtime_s": dt,
            "total_motif_instances": int(counts.sum()),
        }
        for i, v in enumerate(counts, start=1):
            row[f"motif_{i}"] = int(v)
            row[f"cp_{i}"] = float(cp[i - 1])
        rows.append(row)
    motif_df = pd.DataFrame(rows)
    motif_df.to_csv(out_dir / "motif_counts_and_cp.csv", index=False)
    return motif_df, cp_map


def write_summary(
    out_path: Path,
    meta_df: pd.DataFrame,
    basic_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    motif_df: pd.DataFrame,
) -> None:
    lines = [
        "# Uploaded Raw Benchmark Analysis Summary",
        "",
        "## Dataset Meta",
        "",
        meta_df.to_markdown(index=False),
        "",
        "## Basic Stats",
        "",
        basic_df.to_markdown(index=False),
        "",
        "## Pairwise Similarity",
        "",
        pair_df.to_markdown(index=False),
        "",
        "## Motif Runtime",
        "",
        motif_df[["dataset", "domain", "num_nodes", "num_hyperedges", "engine", "runtime_s", "total_motif_instances"]].to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- 这次分析直接读取 `data/benchmark_raw/<dataset>/hyperedges-*.txt`。",
        "- `contact-high-school` 属于原始说明里的目标结构分析域。",
        "- `senate-bills` 不在 MoCHy 论文 11 个 benchmark 名单里，但可以作为补充政治域数据。",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, default="data/benchmark_raw")
    parser.add_argument("--datasets", type=str, default="contact-high-school,senate-bills")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pairs_per_repeat", type=int, default=100000)
    parser.add_argument("--overlap_repeats", type=int, default=3)
    parser.add_argument("--mochy_repo", type=str, default="/home/user/GSK/mgao/MoCHy_ref")
    parser.add_argument("--num_threads", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="outputs_transferability/uploaded_raw_pair_seed7")
    args = parser.parse_args()

    raw_root = ROOT / args.raw_root
    out_dir = ROOT / args.output_dir
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    logs_dir = out_dir / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    raw_graphs = []
    processed_graphs = []
    meta_rows = []
    for name in dataset_names:
        raw_g, proc_g = load_raw_benchmark(raw_root / name)
        raw_graphs.append(raw_g)
        processed_graphs.append(proc_g)
        meta_rows.append(
            {
                "dataset": raw_g.name,
                "domain": raw_g.domain,
                "raw_label_count": raw_g.n_labels,
                "label_name_count": raw_g.n_label_names,
                "post_clean_nodes": raw_g.num_nodes,
                "post_clean_hyperedges": len(raw_g.hyperedges),
            }
        )
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(tables_dir / "dataset_meta.csv", index=False)

    size_map = {g.name: size_features(g) for g in processed_graphs}
    overlap_map = {
        g.name: sample_overlap_features(g, pairs_per_repeat=args.pairs_per_repeat, repeats=args.overlap_repeats, seed=args.seed + i * 17)
        for i, g in enumerate(processed_graphs)
    }
    incidence_map = {g.name: incidence_features(g) for g in processed_graphs}
    community_map = {
        g.name: community_features(g, incidence_map[g.name]["B"], seed=args.seed)
        for g in processed_graphs
    }

    basic_rows = []
    for g in processed_graphs:
        sizes = np.asarray([len(edge) for edge in g.hyperedges], dtype=np.float64)
        node_deg = np.asarray(incidence_map[g.name]["node_degree"], dtype=np.float64)
        basic_rows.append(
            {
                "dataset": g.name,
                "domain": g.domain,
                "num_nodes": g.num_nodes,
                "num_hyperedges": len(g.hyperedges),
                "avg_hyperedge_size": float(sizes.mean()) if sizes.size else 0.0,
                "median_hyperedge_size": float(np.median(sizes)) if sizes.size else 0.0,
                "max_hyperedge_size": int(sizes.max()) if sizes.size else 0,
                "avg_hyperdegree": float(node_deg.mean()) if node_deg.size else 0.0,
                "max_hyperdegree": float(node_deg.max()) if node_deg.size else 0.0,
                "density": float(incidence_map[g.name]["density"]),
                "num_connected_components": int(incidence_map[g.name]["num_connected_components"]),
            }
        )
    basic_df = pd.DataFrame(basic_rows)
    basic_df.to_csv(tables_dir / "dataset_basic_stats.csv", index=False)

    matrices = compute_similarity_matrices(processed_graphs, size_map, overlap_map, incidence_map, community_map)
    for name, df in matrices.items():
        df.to_csv(tables_dir / f"{name}_similarity_matrix.csv")
        save_heatmap(df, figures_dir / f"{name}_similarity_heatmap.png", f"{name.capitalize()} Similarity")
    save_size_distribution(processed_graphs, size_map, figures_dir / "hyperedge_size_distribution.png")

    motif_df, cp_map = motif_counts_for_graphs(processed_graphs, Path(args.mochy_repo), out_dir, num_threads=args.num_threads)
    motif_cos = compute_similarity(cp_map, "cosine")
    motif_pear = compute_similarity(cp_map, "pearson")
    motif_euc = compute_similarity(cp_map, "euclidean_similarity")
    motif_cos.to_csv(tables_dir / "motif_similarity_cosine.csv")
    motif_pear.to_csv(tables_dir / "motif_similarity_pearson.csv")
    motif_euc.to_csv(tables_dir / "motif_similarity_euclidean_similarity.csv")
    save_heatmap(motif_cos, figures_dir / "motif_similarity_cosine.png", "Motif CP Cosine Similarity")
    save_heatmap(motif_pear, figures_dir / "motif_similarity_pearson.png", "Motif CP Pearson Similarity")

    pair_rows = []
    for g1, g2 in combinations(processed_graphs, 2):
        relation = "within" if g1.domain == g2.domain else "cross"
        for factor, df in matrices.items():
            pair_rows.append(
                {
                    "factor": factor,
                    "dataset_a": g1.name,
                    "dataset_b": g2.name,
                    "relation": relation,
                    "similarity": float(df.loc[g1.name, g2.name]),
                }
            )
        pair_rows.append(
            {
                "factor": "motif_cosine",
                "dataset_a": g1.name,
                "dataset_b": g2.name,
                "relation": relation,
                "similarity": float(motif_cos.loc[g1.name, g2.name]),
            }
        )
        pair_rows.append(
            {
                "factor": "motif_pearson",
                "dataset_a": g1.name,
                "dataset_b": g2.name,
                "relation": relation,
                "similarity": float(motif_pear.loc[g1.name, g2.name]),
            }
        )
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(tables_dir / "pair_similarity_summary.csv", index=False)

    write_summary(logs_dir / "uploaded_raw_pair_summary.md", meta_df, basic_df, pair_df, motif_df)
    print(f"[uploaded-raw-analysis] done -> {out_dir}")


if __name__ == "__main__":
    main()
