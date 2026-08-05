#!/usr/bin/env python3
"""
compare_transfer_results.py
Compares pretrained vs scratch downstream results and prints a paper-ready summary table.

Usage:
    python scripts/compare_transfer_results.py
        [--results_dir OUTPUTS_RESULTS_DIR]
        [--output_markdown OUTPUT_FILE]

The script reads all transfer_*.json files under the results directory and
produces a side-by-side comparison grouped by task type.

Output includes:
    - Per-dataset metrics (accuracy / macro_f1 for node, hr@10 / ndcg@10 for rec)
    - Mean ± std across seeds
    - Transfer gain: pretrained - scratch
    - A markdown table ready to paste into a paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pretrained vs scratch downstream results.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="outputs/results",
        help="Directory containing transfer_*.json result files.",
    )
    parser.add_argument(
        "--output_markdown",
        type=str,
        default="",
        help="If set, also write the markdown table to this file.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def parse_transfer_file(path: Path) -> Optional[Dict]:
    """Parse a single transfer_*.json file and return its contents, or None on error."""
    try:
        return load_json(path)
    except Exception:
        return None


def classify_file(path: Path, result: Optional[Dict] = None) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Returns (task, target, is_scratch).
    task: 'node' or 'rec' or None
    target: e.g. 'citation', 'academic', 'gowalla', ... or None
    is_scratch: True if the file comes from a scratch config run
    """
    name = path.stem  # e.g. transfer_node_citation_best
    # Determine scratch: check filename for '_scratch' / '_null', or check JSON for pretrained_checkpoint
    is_scratch = "_scratch" in name or "_null" in name
    if result is not None:
        ckpt = result.get("pretrained_checkpoint") or result.get("training", {}).get("pretrained_checkpoint")
        # If a checkpoint path exists and is not null/empty, it is pretrained
        if ckpt and str(ckpt).strip() not in ("", "null", "None"):
            is_scratch = False
        # Also use pretrain_domains as a signal: if it exists and has entries, the model was pretrained
        if is_scratch and result.get("pretrain_domains"):
            is_scratch = False

    # Try to get heldout_domain directly from JSON result
    heldout = None
    if result is not None:
        heldout = result.get("heldout_domain") or result.get("task") or None

    # Also try to read task from result
    task = None
    if result is not None:
        task_val = result.get("task") or result.get("task_name")
        if task_val in ("node", "rec", "recommendation"):
            task = "node" if task_val == "node" else "rec"

    # Fallback: parse from filename
    if task is None:
        parts = name.replace("transfer_", "").split("_")
        if not parts:
            return None, None, False
        if parts[0] == "node":
            task = "node"
            parts = parts[1:]
        elif parts[0] == "rec":
            task = "rec"
            parts = parts[1:]
        else:
            return None, None, False
        target_parts = [p for p in parts if p not in ("best", "scratch", "null")]
        heldout = "_".join(target_parts) if target_parts else None

    return task, heldout, is_scratch


def extract_node_metrics(result: Dict) -> Dict[str, float]:
    """Extract node classification metrics from a result dict."""
    out = {}
    # Support both 'node_accuracy_mean' (newer) and 'node_accuracy' (existing files)
    out["node_accuracy_mean"] = float(result.get("node_accuracy_mean", result.get("node_accuracy", 0.0)))
    out["node_accuracy_std"] = float(result.get("node_accuracy_std", 0.0))
    out["node_macro_f1_mean"] = float(result.get("node_macro_f1_mean", result.get("node_macro_f1", 0.0)))
    out["node_macro_f1_std"] = float(result.get("node_macro_f1_std", 0.0))
    # Also handle per-dataset results (newer format)
    dataset_results = result.get("dataset_results", [])
    out["datasets"] = []
    for dr in dataset_results:
        ds = {}
        ds["name"] = dr.get("dataset_name", "unknown")
        for key in ("node_accuracy_mean", "node_accuracy_std", "node_macro_f1_mean", "node_macro_f1_std"):
            ds[key] = float(dr.get(key, 0.0))
        if ds["name"] != "unknown":
            out["datasets"].append(ds)
    return out


def extract_rec_metrics(result: Dict) -> Dict[str, float]:
    """Extract recommendation metrics from a result dict."""
    out = {}
    # Support both 'hr@5_mean' (newer) and 'hr@5' (existing files)
    out["hr@10_mean"] = float(result.get("hr@10_mean", result.get("hr@10", 0.0)))
    out["hr@10_std"] = float(result.get("hr@10_std", 0.0))
    out["ndcg@10_mean"] = float(result.get("ndcg@10_mean", result.get("ndcg@10", 0.0)))
    out["ndcg@10_std"] = float(result.get("ndcg@10_std", 0.0))
    out["hr@5_mean"] = float(result.get("hr@5_mean", result.get("hr@5", 0.0)))
    out["hr@5_std"] = float(result.get("hr@5_std", 0.0))
    out["ndcg@5_mean"] = float(result.get("ndcg@5_mean", result.get("ndcg@5", 0.0)))
    out["ndcg@5_std"] = float(result.get("ndcg@5_std", 0.0))
    # Use evaluated_datasets as the key for rec (heldout_domain is the domain, not specific dataset)
    dataset_results = result.get("dataset_results", [])
    out["datasets"] = []
    for dr in dataset_results:
        ds = {}
        ds["name"] = dr.get("dataset_name", "unknown")
        if ds["name"] == "unknown":
            continue
        for key in ("hr@5_mean", "hr@5_std", "hr@10_mean", "hr@10_std", "ndcg@5_mean", "ndcg@5_std", "ndcg@10_mean", "ndcg@10_std"):
            ds[key] = float(dr.get(key, dr.get(key.replace("_mean", ""), 0.0)))
        out["datasets"].append(ds)
    return out


def fmt(value: float, decimals: int = 4) -> str:
    """Format a float to given decimal places, trimming trailing zeros."""
    return f"{value:.{decimals}f}"


def fmt_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    return f"{fmt(mean, decimals)} ± {fmt(std, decimals)}"


def fmt_gain(gain: float, decimals: int = 4) -> str:
    sign = "+" if gain >= 0 else ""
    return f"{sign}{fmt(gain, decimals)}"


def build_node_table(
    pretrained: Dict[str, Dict],  # {dataset_name: {metric_name: value}}
    scratch: Dict[str, Dict],
) -> List[str]:
    lines = []
    lines.append("## Node Classification")
    lines.append("")
    lines.append("| Dataset | Method | Accuracy | Macro-F1 |")
    lines.append("|---------|--------|----------|----------|")

    # Collect all dataset names from both buckets
    all_datasets = sorted(set(list(pretrained.keys()) + list(scratch.keys())))
    for ds in all_datasets:
        pre = pretrained.get(ds, {})
        scr = scratch.get(ds, {})
        pre_acc = pre.get("node_accuracy_mean", 0.0)
        pre_std = pre.get("node_accuracy_std", 0.0)
        scr_acc = scr.get("node_accuracy_mean", 0.0)
        scr_std = scr.get("node_accuracy_std", 0.0)
        pre_f1 = pre.get("node_macro_f1_mean", 0.0)
        pre_f1_std = pre.get("node_macro_f1_std", 0.0)
        scr_f1 = scr.get("node_macro_f1_mean", 0.0)
        scr_f1_std = scr.get("node_macro_f1_std", 0.0)
        if pre_acc == 0 and scr_acc == 0:
            continue
        lines.append(
            f"| **{ds}** | Ours (pretrained) | "
            f"{fmt_mean_std(pre_acc, pre_std)} | "
            f"{fmt_mean_std(pre_f1, pre_f1_std)} |"
        )
        lines.append(
            f"| | Scratch (random init) | "
            f"{fmt_mean_std(scr_acc, scr_std)} | "
            f"{fmt_mean_std(scr_f1, scr_f1_std)} |"
        )
        acc_gain = pre_acc - scr_acc
        f1_gain = pre_f1 - scr_f1
        lines.append(
            f"| | **Transfer Gain** | "
            f"**{fmt_gain(acc_gain)}** | "
            f"**{fmt_gain(f1_gain)}** |"
        )
        lines.append("| | | | |")
    return lines


def build_rec_table(
    pretrained: Dict[str, Dict],
    scratch: Dict[str, Dict],
) -> List[str]:
    lines = []
    lines.append("## Recommendation Tasks")
    lines.append("")
    lines.append("| Dataset | Method | HR@10 | NDCG@10 |")
    lines.append("|---------|--------|-------|----------|")

    dataset_order = ["gowalla", "yelp_2018", "movielens_1m"]
    for ds in dataset_order:
        pre = pretrained.get(ds, {})
        scr = scratch.get(ds, {})
        pre_hr = pre.get("hr@10_mean", 0.0)
        pre_hr_std = pre.get("hr@10_std", 0.0)
        scr_hr = scr.get("hr@10_mean", 0.0)
        scr_hr_std = scr.get("hr@10_std", 0.0)
        pre_ndcg = pre.get("ndcg@10_mean", 0.0)
        pre_ndcg_std = pre.get("ndcg@10_std", 0.0)
        scr_ndcg = scr.get("ndcg@10_mean", 0.0)
        scr_ndcg_std = scr.get("ndcg@10_std", 0.0)
        if pre_hr == 0 and scr_hr == 0:
            continue
        lines.append(
            f"| **{ds}** | Ours (pretrained) | "
            f"{fmt_mean_std(pre_hr, pre_hr_std)} | "
            f"{fmt_mean_std(pre_ndcg, pre_ndcg_std)} |"
        )
        lines.append(
            f"| | Scratch (random init) | "
            f"{fmt_mean_std(scr_hr, scr_hr_std)} | "
            f"{fmt_mean_std(scr_ndcg, scr_ndcg_std)} |"
        )
        hr_gain = pre_hr - scr_hr
        ndcg_gain = pre_ndcg - scr_ndcg
        lines.append(
            f"| | **Transfer Gain** | "
            f"**{fmt_gain(hr_gain)}** | "
            f"**{fmt_gain(ndcg_gain)}** |"
        )
        lines.append("| | | | |")
    return lines


def build_summary_row(
    pretrained: Dict[str, Dict],
    scratch: Dict[str, Dict],
    metric: str,
) -> Tuple[str, str, str]:
    """Compute mean across datasets. Support both new format (top-level mean) and old format (per-dataset values)."""
    # Support both 'node_accuracy_mean' (new) and 'node_accuracy' (existing files)
    metric_key = metric
    # Try top-level first (new format)
    pre_top = next((v for v in pretrained.values() if v.get(metric_key, 0.0) > 0 or v.get(metric_key) == 0.0), None)
    if pre_top is not None and metric_key in pre_top:
        pre_mean = pre_top[metric_key]
    else:
        pre_vals = []
        for v in pretrained.values():
            val = v.get(metric_key, 0.0)
            if val > 0 or metric_key in v:
                pre_vals.append(val)
        if not pre_vals:
            return "N/A", "N/A", "N/A"
        pre_mean = sum(pre_vals) / len(pre_vals)

    scr_top = next((v for v in scratch.values() if v.get(metric_key, 0.0) > 0 or v.get(metric_key) == 0.0), None)
    if scr_top is not None and metric_key in scr_top:
        scr_mean = scr_top[metric_key]
    else:
        scr_vals = []
        for v in scratch.values():
            val = v.get(metric_key, 0.0)
            if val > 0 or metric_key in v:
                scr_vals.append(val)
        if not scr_vals:
            return "N/A", "N/A", "N/A"
        scr_mean = sum(scr_vals) / len(scr_vals)

    gain = pre_mean - scr_mean
    return (
        fmt_mean_std(pre_mean, 0.0),
        fmt_mean_std(scr_mean, 0.0),
        fmt_gain(gain),
    )


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect all transfer_*.json files
    files = sorted(results_dir.glob("transfer_*.json"))
    if not files:
        print(f"No transfer_*.json files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} result files:")
    for f in files:
        print(f"  {f.name}")

    # Parse and classify
    node_pretrained: Dict[str, Dict] = {}
    node_scratch: Dict[str, Dict] = {}
    rec_pretrained: Dict[str, Dict] = {}
    rec_scratch: Dict[str, Dict] = {}

    for fp in files:
        result = parse_transfer_file(fp)
        if result is None:
            print(f"  [WARN] Could not parse: {fp.name}", file=sys.stderr)
            continue
        task, target, is_scratch = classify_file(fp, result)
        if task is None:
            print(f"  [WARN] Unknown file format: {fp.name}", file=sys.stderr)
            continue

        if task == "node":
            metrics = extract_node_metrics(result)
            # If we have per-dataset results, register each dataset separately
            # If not, register under the heldout domain key
            bucket = node_scratch if is_scratch else node_pretrained
            if metrics.get("datasets"):
                for ds in metrics["datasets"]:
                    bucket[ds["name"]] = ds
            else:
                # Fallback: register under heldout domain with top-level metrics
                bucket[target or "unknown"] = metrics
        elif task == "rec":
            metrics = extract_rec_metrics(result)
            bucket = rec_scratch if is_scratch else rec_pretrained
            if metrics.get("datasets"):
                for ds in metrics["datasets"]:
                    bucket[ds["name"]] = ds
            else:
                bucket[target or "unknown"] = metrics

    # Build output
    lines: List[str] = []
    lines.append("# Downstream Transfer Evaluation Results")
    lines.append("")
    lines.append("**Note:** `Ours (pretrained)` uses the pretrained checkpoint from cross-domain pre-training. "
                 "`Scratch (random init)` uses the same architecture with random initialization.")
    lines.append("All metrics are reported as **mean ± std** across multiple random seeds.")
    lines.append("")

    # Node tables
    if node_pretrained or node_scratch:
        node_lines = build_node_table(node_pretrained, node_scratch)
        lines.extend(node_lines)
        lines.append("")

        # Summary row for node
        lines.append("### Node Classification: Summary (average across datasets)")
        lines.append("")
        lines.append("| Metric | Ours (pretrained) | Scratch | Transfer Gain |")
        lines.append("|--------|-------------------|--------|---------------|")
        pre_acc, scr_acc, acc_gain = build_summary_row(
            node_pretrained, node_scratch, "node_accuracy_mean"
        )
        pre_f1, scr_f1, f1_gain = build_summary_row(
            node_pretrained, node_scratch, "node_macro_f1_mean"
        )
        lines.append(f"| Accuracy | {pre_acc} | {scr_acc} | **{acc_gain}** |")
        lines.append(f"| Macro-F1 | {pre_f1} | {scr_f1} | **{f1_gain}** |")
        lines.append("")

    # Rec tables
    if rec_pretrained or rec_scratch:
        rec_lines = build_rec_table(rec_pretrained, rec_scratch)
        lines.extend(rec_lines)
        lines.append("")

        lines.append("### Recommendation: Summary (average across datasets)")
        lines.append("")
        lines.append("| Metric | Ours (pretrained) | Scratch | Transfer Gain |")
        lines.append("|--------|-------------------|--------|---------------|")
        pre_hr, scr_hr, hr_gain = build_summary_row(
            rec_pretrained, rec_scratch, "hr@10_mean"
        )
        pre_ndcg, scr_ndcg, ndcg_gain = build_summary_row(
            rec_pretrained, rec_scratch, "ndcg@10_mean"
        )
        lines.append(f"| HR@10 | {pre_hr} | {scr_hr} | **{hr_gain}** |")
        lines.append(f"| NDCG@10 | {pre_ndcg} | {scr_ndcg} | **{ndcg_gain}** |")
        lines.append("")

    output = "\n".join(lines)
    print("\n" + "=" * 80)
    print(output)
    print("=" * 80)

    if args.output_markdown:
        out_path = Path(args.output_markdown)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        print(f"\nMarkdown table written to: {out_path}")


if __name__ == "__main__":
    main()
