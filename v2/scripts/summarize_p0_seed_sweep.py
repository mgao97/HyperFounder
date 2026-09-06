from __future__ import annotations

import csv
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATASETS = ["cora_cc", "citeseer_cc", "pubmed_cc", "coauthorship_dblp", "cooking_200"]


def load_probe_csv(path: Path) -> dict:
    rows = list(csv.DictReader(path.open()))
    by_ds = {}
    for r in rows:
        ds = r["dataset"]
        by_ds.setdefault(ds, []).append(float(r["delta_pp"]))
    ds_mean = {k: sum(v) / len(v) for k, v in by_ds.items()}
    grand = sum(ds_mean.values()) / len(ds_mean)
    base_mean = {}
    ours_mean = {}
    for ds in DATASETS:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        base_mean[ds] = sum(float(r["baseline_acc"]) for r in ds_rows) / len(ds_rows) * 100.0
        ours_mean[ds] = sum(float(r["ours_acc"]) for r in ds_rows) / len(ds_rows) * 100.0
    return {
        "rows": rows,
        "ds_mean": ds_mean,
        "grand": grand,
        "base_mean": base_mean,
        "ours_mean": ours_mean,
    }


def mean_std(vals: list[float]) -> tuple[float, float]:
    if len(vals) == 1:
        return vals[0], 0.0
    return st.mean(vals), st.stdev(vals)


def fmt(ms: tuple[float, float], signed: bool = True) -> str:
    m, s = ms
    if signed:
        return f"{m:+.2f} ± {s:.2f}"
    return f"{m:.2f} ± {s:.2f}"


def main() -> None:
    sources = {
        "full": {
            42: ROOT / "outputs_v2/ablations_seed42/w3_full/lodo_probe.csv",
            1: ROOT / "outputs_v2/p0_seed_sweep/seed_1/full/lodo_probe.csv",
            7: ROOT / "outputs_v2/p0_seed_sweep/seed_7/full/lodo_probe.csv",
        },
        "no_card": {
            42: ROOT / "outputs_v2/ablations_seed42/w3_no_card/lodo_probe.csv",
            1: ROOT / "outputs_v2/p0_seed_sweep/seed_1/no_card/lodo_probe.csv",
            7: ROOT / "outputs_v2/p0_seed_sweep/seed_7/no_card/lodo_probe.csv",
        },
        "no_HCA": {
            42: ROOT / "outputs_v2/ablations_seed42/w4_no_hca/lodo_probe.csv",
            1: ROOT / "outputs_v2/p0_seed_sweep/seed_1/no_HCA/lodo_probe.csv",
            7: ROOT / "outputs_v2/p0_seed_sweep/seed_7/no_HCA/lodo_probe.csv",
        },
        "no_HOR": {
            42: ROOT / "outputs_v2/ablations_seed42/w5_without_hor/lodo_probe.csv",
            1: ROOT / "outputs_v2/p0_seed_sweep/seed_1/no_HOR/lodo_probe.csv",
            7: ROOT / "outputs_v2/p0_seed_sweep/seed_7/no_HOR/lodo_probe.csv",
        },
    }
    scratch_path = ROOT / "outputs_v2/p0_seed_sweep/scratch/lodo_probe.csv"

    loaded: dict[str, dict[int, dict]] = {}
    for variant, mapping in sources.items():
        loaded[variant] = {}
        for seed, path in mapping.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {variant} seed={seed}: {path}")
            loaded[variant][seed] = load_probe_csv(path)

    scratch = load_probe_csv(scratch_path) if scratch_path.is_file() else None

    out_lines: list[str] = []
    out_lines.append("# P0-1 pretrain seed sweep summary")
    out_lines.append("")
    out_lines.append("## Grand mean Δ across pretrain seeds {42,1,7}")
    out_lines.append("")
    out_lines.append("| variant | seed42 | seed1 | seed7 | mean±std (pretrain seeds) |")
    out_lines.append("|---|---:|---:|---:|---:|")

    gm_by_variant: dict[str, tuple[float, float]] = {}
    for variant in ("full", "no_card", "no_HCA", "no_HOR"):
        vals = [loaded[variant][s]["grand"] for s in (42, 1, 7)]
        gm_by_variant[variant] = mean_std(vals)
        out_lines.append(
            f"| {variant} | {vals[0]:+.2f} | {vals[1]:+.2f} | {vals[2]:+.2f} | {fmt(gm_by_variant[variant])} |"
        )

    out_lines.append("")
    out_lines.append("## Relative to full (same pretrain seed)")
    out_lines.append("")
    out_lines.append("| baseline | seed42 | seed1 | seed7 | mean±std |")
    out_lines.append("|---|---:|---:|---:|---:|")
    rel_stats: dict[str, tuple[float, float]] = {}
    for variant in ("no_card", "no_HCA", "no_HOR"):
        vals = [
            loaded[variant][s]["grand"] - loaded["full"][s]["grand"]
            for s in (42, 1, 7)
        ]
        rel_stats[variant] = mean_std(vals)
        out_lines.append(
            f"| {variant} - full | {vals[0]:+.2f} | {vals[1]:+.2f} | {vals[2]:+.2f} | {fmt(rel_stats[variant])} |"
        )

    if scratch is not None:
        out_lines.append("")
        out_lines.append("## Scratch encoder control")
        out_lines.append("")
        out_lines.append(f"- scratch encoder Grand mean Δ = {scratch['grand']:+.2f} pp")

    # Main P0-1 criterion against strongest baseline.
    strongest_name = max(("no_card", "no_HCA", "no_HOR"), key=lambda k: gm_by_variant[k][0])
    full_m, full_s = gm_by_variant["full"]
    base_m, base_s = gm_by_variant[strongest_name]
    combined_std = math.sqrt(full_s ** 2 + base_s ** 2)
    passes = (full_m - base_m) > (2.0 * combined_std)

    out_lines.append("")
    out_lines.append("## P0-1 verdict")
    out_lines.append("")
    out_lines.append(f"- strongest baseline by mean Δ = `{strongest_name}` with {fmt(gm_by_variant[strongest_name])} pp")
    out_lines.append(f"- full method = {fmt(gm_by_variant['full'])} pp")
    out_lines.append(f"- criterion: full - strongest_baseline > 2 * pooled_std = {2.0 * combined_std:.2f} pp")
    out_lines.append(f"- observed margin = {full_m - base_m:+.2f} pp")
    out_lines.append(f"- **PASS = {'YES' if passes else 'NO'}**")

    out_lines.append("")
    out_lines.append("## Per-dataset full mean±std across pretrain seeds")
    out_lines.append("")
    out_lines.append("| dataset | full Δ mean±std |")
    out_lines.append("|---|---:|")
    for ds in DATASETS:
        vals = [loaded["full"][s]["ds_mean"][ds] for s in (42, 1, 7)]
        out_lines.append(f"| {ds} | {fmt(mean_std(vals))} |")

    target = ROOT / "logs" / "p0_seed_sweep_summary.md"
    target.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print("\n".join(out_lines))
    print(f"\n[summary] wrote {target}")


if __name__ == "__main__":
    main()
