from __future__ import annotations

import argparse
import csv
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        raise ValueError("empty values")
    if len(vals) == 1:
        return vals[0], 0.0
    return st.mean(vals), st.stdev(vals)


def _load_by_dataset(path: Path, metric_key: str) -> dict[str, list[float]]:
    rows = list(csv.DictReader(path.open()))
    out: dict[str, list[float]] = {}
    for row in rows:
        out.setdefault(row["dataset"], []).append(float(row[metric_key]))
    return out


def _dataset_delta_stats(path: Path, metric_key: str) -> tuple[dict[str, float], tuple[float, float]]:
    by_ds = _load_by_dataset(path, metric_key)
    ds_mean = {ds: sum(vals) / len(vals) for ds, vals in by_ds.items()}
    grand = _mean_std(list(ds_mean.values()))
    return ds_mean, grand


def _fmt(ms: tuple[float, float], signed: bool = True) -> str:
    mean, std = ms
    if signed:
        return f"{mean:+.2f} ± {std:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def _verdict(full_stats: tuple[float, float], scratch_stats: tuple[float, float]) -> tuple[bool, float, float]:
    full_m, full_s = full_stats
    scratch_m, scratch_s = scratch_stats
    pooled = math.sqrt(full_s ** 2 + scratch_s ** 2)
    margin = full_m - scratch_m
    threshold = 2.0 * pooled
    return margin > threshold, margin, threshold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_dir",
        type=str,
        default="outputs_v2/p0_frozen_vs_finetune",
        help="Directory containing frozen_full/scratch and finetune_full/scratch CSVs.",
    )
    ap.add_argument(
        "--out_md",
        type=str,
        default="logs/s0_main_contribution_summary.md",
        help="Markdown report path.",
    )
    args = ap.parse_args()

    in_dir = ROOT / args.in_dir if not Path(args.in_dir).is_absolute() else Path(args.in_dir)
    out_md = ROOT / args.out_md if not Path(args.out_md).is_absolute() else Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    files = {
        "frozen_full": in_dir / "frozen_full.csv",
        "frozen_scratch": in_dir / "frozen_scratch.csv",
        "finetune_full": in_dir / "finetune_full.csv",
        "finetune_scratch": in_dir / "finetune_scratch.csv",
    }
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing required file for {name}: {path}")

    frozen_full_ds, frozen_full_stats = _dataset_delta_stats(files["frozen_full"], "delta_pp")
    frozen_scratch_ds, frozen_scratch_stats = _dataset_delta_stats(files["frozen_scratch"], "delta_pp")
    finetune_full_ds, finetune_full_stats = _dataset_delta_stats(files["finetune_full"], "delta_vs_raw_pp")
    finetune_scratch_ds, finetune_scratch_stats = _dataset_delta_stats(files["finetune_scratch"], "delta_vs_raw_pp")

    frozen_pass, frozen_margin, frozen_thr = _verdict(frozen_full_stats, frozen_scratch_stats)
    finetune_pass, finetune_margin, finetune_thr = _verdict(finetune_full_stats, finetune_scratch_stats)

    lines: list[str] = []
    lines.append("# S0-1 Main Contribution Check")
    lines.append("")
    lines.append("Criterion: `full mean - scratch mean > 2 x pooled_std`, where mean/std are computed")
    lines.append("over the 5 W1/W2 LODO task-level deltas (dataset means after averaging eval seeds).")
    lines.append("")
    lines.append("## Protocol Summary")
    lines.append("")
    lines.append("| protocol | full Δ mean±std (pp) | scratch Δ mean±std (pp) | margin (pp) | threshold (2x pooled std) | pass |")
    lines.append("|---|---:|---:|---:|---:|---|")
    lines.append(
        f"| frozen | {_fmt(frozen_full_stats)} | {_fmt(frozen_scratch_stats)} | "
        f"{frozen_margin:+.2f} | {frozen_thr:.2f} | {'YES' if frozen_pass else 'NO'} |"
    )
    lines.append(
        f"| finetune | {_fmt(finetune_full_stats)} | {_fmt(finetune_scratch_stats)} | "
        f"{finetune_margin:+.2f} | {finetune_thr:.2f} | {'YES' if finetune_pass else 'NO'} |"
    )

    lines.append("")
    lines.append("## Task-level Deltas")
    lines.append("")
    lines.append("| dataset | frozen full | frozen scratch | finetune full | finetune scratch |")
    lines.append("|---|---:|---:|---:|---:|")
    datasets = sorted(set(frozen_full_ds) | set(frozen_scratch_ds) | set(finetune_full_ds) | set(finetune_scratch_ds))
    for ds in datasets:
        lines.append(
            f"| {ds} | {frozen_full_ds[ds]:+.2f} | {frozen_scratch_ds[ds]:+.2f} | "
            f"{finetune_full_ds[ds]:+.2f} | {finetune_scratch_ds[ds]:+.2f} |"
        )

    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if frozen_pass:
        lines.append("- Frozen protocol supports the main contribution under the pre-registered criterion.")
    else:
        lines.append("- Frozen protocol does **not** support the main contribution under the pre-registered criterion.")
    if finetune_pass:
        lines.append("- Finetune protocol also supports the main contribution under the same criterion.")
    else:
        lines.append("- Finetune protocol does **not** support the main contribution under the same criterion.")

    if frozen_pass and finetune_pass:
        lines.append("- Overall S0-1 verdict: `main contribution holds in both protocols`.")
    elif frozen_pass and not finetune_pass:
        lines.append("- Overall S0-1 verdict: `main contribution only holds in frozen protocol`.")
    else:
        lines.append("- Overall S0-1 verdict: `main contribution claim must be weakened or reframed`.")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[summary] wrote {out_md}")


if __name__ == "__main__":
    main()
