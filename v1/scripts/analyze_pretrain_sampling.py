from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.common import load_yaml, save_json, set_seed
from utils.dhg_datasets import load_domain_graphs
from utils.hypergraph import SimpleHypergraph, iter_graphs
from utils.minibatch_sampling import (
    build_subhypergraph_pool,
    sample_online_subhypergraph,
    sample_subhypergraph_batch,
    should_use_subhypergraph_pool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze dataset statistics and subhypergraph sampling stability.")
    parser.add_argument("--config", required=True, help="Path to pretrain config yaml.")
    parser.add_argument("--num_samples", type=int, default=64, help="How many subhypergraphs to sample for analysis.")
    parser.add_argument("--batch_mode", action="store_true", help="Use sample_subhypergraph_batch instead of per-graph online sampling.")
    parser.add_argument("--use_pool_cache", action="store_true", help="Build and use subhypergraph pool cache for large graphs.")
    parser.add_argument("--scan_pool_eigh", action="store_true", help="When using pool cache, scan all pooled subhypergraphs with eigh.")
    parser.add_argument("--scan_pool_limit", type=int, default=0, help="Optional limit on pooled subhypergraphs per graph (0 means no limit).")
    parser.add_argument("--device", default="cpu", help="cpu or cuda.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def _edge_sizes(hg: SimpleHypergraph) -> np.ndarray:
    if not hg.hyperedges:
        return np.asarray([], dtype=np.int64)
    return np.asarray([len(edge) for edge in hg.hyperedges], dtype=np.int64)


def _node_degrees_from_edges(hg: SimpleHypergraph) -> np.ndarray:
    degrees = np.zeros((hg.num_nodes,), dtype=np.int64)
    for edge in hg.hyperedges:
        for node_id in edge:
            if 0 <= node_id < hg.num_nodes:
                degrees[node_id] += 1
    return degrees


def _summarize_array(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0, "mean": 0.0}
    quantiles = np.quantile(values.astype(np.float64), [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "min": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "max": float(quantiles[4]),
        "mean": float(np.mean(values.astype(np.float64))),
    }


def summarize_graph(hg: SimpleHypergraph) -> Dict[str, Any]:
    edge_sizes = _edge_sizes(hg)
    node_degrees = _node_degrees_from_edges(hg)
    return {
        "name": hg.name,
        "dataset_name": hg.dataset_name,
        "domain": hg.domain,
        "num_nodes": int(hg.num_nodes),
        "num_hyperedges": int(len(hg.hyperedges)),
        "edge_size": _summarize_array(edge_sizes),
        "node_degree": _summarize_array(node_degrees),
    }


def _unique_edge_count(hg: SimpleHypergraph) -> int:
    if not hg.hyperedges:
        return 0
    unique = {tuple(edge) for edge in (tuple(sorted(set(e))) for e in hg.hyperedges if e)}
    return len(unique)


def analyze_subhypergraph(hg: SimpleHypergraph, device: torch.device) -> Dict[str, Any]:
    incidence = hg.incidence_matrix().to(device)
    node_overlap = incidence @ incidence.transpose(0, 1)
    sym_matrix = (node_overlap + node_overlap.transpose(0, 1)) * 0.5
    sym_matrix = torch.nan_to_num(sym_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    sym_cpu = sym_matrix.detach().float().cpu()
    stats: Dict[str, Any] = {
        "dataset_name": hg.dataset_name,
        "domain": hg.domain,
        "name": hg.name,
        "num_nodes": int(hg.num_nodes),
        "num_hyperedges": int(len(hg.hyperedges)),
        "unique_hyperedges": int(_unique_edge_count(hg)),
        "incidence_nnz": int((incidence != 0).sum().item()),
        "node_overlap_diag_min": float(torch.diagonal(node_overlap).min().item()) if node_overlap.numel() else 0.0,
        "node_overlap_diag_max": float(torch.diagonal(node_overlap).max().item()) if node_overlap.numel() else 0.0,
        "symmetry_error_fro": float(torch.linalg.norm(sym_matrix - sym_matrix.transpose(0, 1)).item())
        if sym_matrix.numel()
        else 0.0,
    }
    metadata = hg.metadata or {}
    for key in ("parent_graph_name", "parent_dataset_name", "global_node_ids", "global_edge_ids", "seed_edge_ids", "sampling_depth"):
        if key in metadata:
            stats[key] = metadata[key]
    try:
        eigvals, _ = torch.linalg.eigh(sym_matrix)
        eigvals = torch.nan_to_num(eigvals, nan=0.0, posinf=0.0, neginf=0.0)
        stats["eigh_ok"] = True
        stats["eig_min"] = float(eigvals.min().item()) if eigvals.numel() else 0.0
        stats["eig_max"] = float(eigvals.max().item()) if eigvals.numel() else 0.0
        stats["eig_zero_count"] = int((eigvals.abs() < 1e-8).sum().item()) if eigvals.numel() else 0
        stats["rank"] = int(torch.linalg.matrix_rank(sym_cpu).item()) if sym_cpu.numel() else 0
    except Exception as exc:
        stats["eigh_ok"] = False
        stats["eigh_error"] = repr(exc)
        try:
            stats["rank"] = int(torch.linalg.matrix_rank(sym_cpu).item()) if sym_cpu.numel() else 0
        except Exception as rank_exc:
            stats["rank"] = None
            stats["rank_error"] = repr(rank_exc)
        try:
            eigvals_cpu, _ = torch.linalg.eigh(sym_cpu)
            eigvals_cpu = torch.nan_to_num(eigvals_cpu, nan=0.0, posinf=0.0, neginf=0.0)
            stats["eigh_cpu_ok"] = True
            stats["eig_cpu_min"] = float(eigvals_cpu.min().item()) if eigvals_cpu.numel() else 0.0
            stats["eig_cpu_max"] = float(eigvals_cpu.max().item()) if eigvals_cpu.numel() else 0.0
            stats["eig_cpu_zero_count"] = int((eigvals_cpu.abs() < 1e-8).sum().item()) if eigvals_cpu.numel() else 0
        except Exception as cpu_exc:
            stats["eigh_cpu_ok"] = False
            stats["eigh_cpu_error"] = repr(cpu_exc)
    return stats


def _scan_pool_cache(
    pool_cache: Dict[str, List[SimpleHypergraph]],
    device: torch.device,
    per_graph_limit: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    scanned: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    summaries: Dict[str, Dict[str, Any]] = {}
    for parent_name, pool in pool_cache.items():
        limit = per_graph_limit if per_graph_limit > 0 else len(pool)
        checked = 0
        graph_failures = 0
        worst: Dict[str, Any] | None = None
        for subhg in pool[:limit]:
            try:
                stats = analyze_subhypergraph(subhg, device=device)
            except Exception as exc:
                stats = {
                    "dataset_name": subhg.dataset_name,
                    "domain": subhg.domain,
                    "name": subhg.name,
                    "num_nodes": int(subhg.num_nodes),
                    "num_hyperedges": int(len(subhg.hyperedges)),
                    "eigh_ok": False,
                    "fatal_error": repr(exc),
                    "parent_graph_name": subhg.metadata.get("parent_graph_name") if subhg.metadata else None,
                    "parent_dataset_name": subhg.metadata.get("parent_dataset_name") if subhg.metadata else None,
                    "seed_edge_ids": subhg.metadata.get("seed_edge_ids") if subhg.metadata else None,
                    "sampling_depth": subhg.metadata.get("sampling_depth") if subhg.metadata else None,
                }
            scanned.append(stats)
            checked += 1
            if not stats["eigh_ok"]:
                failures.append(stats)
                graph_failures += 1
                if worst is None:
                    worst = stats
        summaries[parent_name] = {
            "pool_size": int(len(pool)),
            "checked": int(checked),
            "failures": int(graph_failures),
            "first_failure": worst,
        }
    return scanned, failures, summaries


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    config.setdefault("training", {})
    config["training"]["seed"] = int(args.seed)
    set_seed(int(args.seed))

    device = torch.device(args.device)
    minibatch_config = dict(config["training"].get("minibatch", {}))

    start = time.perf_counter()
    domains = load_domain_graphs(config, seed=int(args.seed))
    graphs = iter_graphs(domains)
    pool_cache: Dict[str, List[SimpleHypergraph]] = {}
    if args.use_pool_cache:
        for graph_index, hg in enumerate(graphs):
            if not should_use_subhypergraph_pool(hg, minibatch_config):
                continue
            pool_cache[hg.name] = build_subhypergraph_pool(
                hg,
                minibatch_config=minibatch_config,
                seed=int(args.seed) + graph_index * 1009,
            )
        pooled = sum(len(pool) for pool in pool_cache.values())
        print(f"[Analyze] Built pool_cache for {len(pool_cache)} graphs with {pooled} subhypergraphs", flush=True)

    pool_scan_stats: List[Dict[str, Any]] = []
    pool_scan_failures: List[Dict[str, Any]] = []
    pool_scan_summaries: Dict[str, Dict[str, Any]] = {}
    if args.use_pool_cache and args.scan_pool_eigh and pool_cache:
        pool_scan_stats, pool_scan_failures, pool_scan_summaries = _scan_pool_cache(
            pool_cache,
            device=device,
            per_graph_limit=int(args.scan_pool_limit),
        )
        print(
            f"[Analyze] Pool eigh scan: checked={len(pool_scan_stats)} failures={len(pool_scan_failures)}",
            flush=True,
        )
        bad_graphs = sorted(
            (name for name, summary in pool_scan_summaries.items() if summary.get("failures", 0) > 0),
            key=lambda name: int(pool_scan_summaries[name]["failures"]),
            reverse=True,
        )
        for name in bad_graphs[:10]:
            summary = pool_scan_summaries[name]
            first = summary.get("first_failure") or {}
            first_error = first.get("eigh_error") or first.get("fatal_error") or first.get("rank_error") or None
            print(
                f"[PoolFailure] parent={name} failures={summary.get('failures')}/{summary.get('checked')} "
                f"first_nodes={first.get('num_nodes')} first_edges={first.get('num_hyperedges')} "
                f"first_unique_edges={first.get('unique_hyperedges')} first_rank={first.get('rank')} "
                f"error={first_error}",
                flush=True,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    print("[Analyze] Loaded domains:", sorted(domains), flush=True)
    print("[Analyze] Total graphs:", len(graphs), flush=True)

    dataset_stats = [summarize_graph(hg) for hg in graphs]
    for item in dataset_stats:
        print(
            "[Dataset]"
            f" {item['dataset_name']}/{item['domain']}"
            f" nodes={item['num_nodes']} edges={item['num_hyperedges']}"
            f" edge_size_median={item['edge_size']['median']:.1f}"
            f" degree_median={item['node_degree']['median']:.1f}",
            flush=True,
        )

    sampled_stats: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    if args.batch_mode:
        steps = max(1, int(np.ceil(args.num_samples / max(int(minibatch_config.get("domains_per_step", 2)), 1))))
        for step in range(steps):
            batch = sample_subhypergraph_batch(
                domains,
                minibatch_config=minibatch_config,
                pool_cache=pool_cache,
                seed=int(args.seed) + step * 1009,
                preferred_domains=None,
            )
            for hg in batch:
                try:
                    stats = analyze_subhypergraph(hg, device=device)
                except Exception as exc:
                    stats = {
                        "dataset_name": hg.dataset_name,
                        "domain": hg.domain,
                        "name": hg.name,
                        "num_nodes": int(hg.num_nodes),
                        "num_hyperedges": int(len(hg.hyperedges)),
                        "eigh_ok": False,
                        "fatal_error": repr(exc),
                    }
                sampled_stats.append(stats)
                if not stats["eigh_ok"]:
                    failures.append(stats)
                if len(sampled_stats) >= args.num_samples:
                    break
            if len(sampled_stats) >= args.num_samples:
                break
    else:
        generator = torch.Generator().manual_seed(int(args.seed))
        for index in range(args.num_samples):
            graph_index = int(torch.randint(0, max(len(graphs), 1), (1,), generator=generator).item())
            parent = graphs[graph_index]
            subhg = sample_online_subhypergraph(parent, minibatch_config=minibatch_config, seed=int(args.seed) + index * 97)
            stats = analyze_subhypergraph(subhg, device=device)
            sampled_stats.append(stats)
            if not stats["eigh_ok"]:
                failures.append(stats)

    print(f"[Analyze] Sampled subhypergraphs: {len(sampled_stats)}", flush=True)
    print(f"[Analyze] eigh failures: {len(failures)}", flush=True)

    for idx, fail in enumerate(failures[:10]):
        print(
            f"[Failure {idx}] {fail.get('dataset_name','?')}/{fail.get('domain','?')} "
            f"nodes={fail.get('num_nodes')} edges={fail.get('num_hyperedges')} "
            f"unique_edges={fail.get('unique_hyperedges')} rank={fail.get('rank')} "
            f"seed_edges={fail.get('seed_edge_ids','-')} error={fail.get('eigh_error','-')}",
            flush=True,
        )

    payload = {
        "config": str(Path(args.config)),
        "seed": int(args.seed),
        "device": str(device),
        "minibatch_config": minibatch_config,
        "dataset_stats": dataset_stats,
        "sampled_stats": sampled_stats,
        "failures": failures,
        "pool_scan": {
            "enabled": bool(args.use_pool_cache and args.scan_pool_eigh),
            "per_graph_limit": int(args.scan_pool_limit),
            "checked": int(len(pool_scan_stats)),
            "failures": int(len(pool_scan_failures)),
            "summaries": pool_scan_summaries,
            "first_failures": pool_scan_failures[:50],
        },
        "elapsed_sec": float(time.perf_counter() - start),
    }
    if args.output:
        save_json(args.output, payload)
        print(f"[Analyze] Wrote JSON: {args.output}", flush=True)
    else:
        default = f"outputs/results/sampling_diagnostics_{int(time.time())}.json"
        save_json(default, payload)
        print(f"[Analyze] Wrote JSON: {default}", flush=True)


if __name__ == "__main__":
    main()
