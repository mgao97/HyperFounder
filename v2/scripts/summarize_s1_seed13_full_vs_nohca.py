from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["cora_cc", "citeseer_cc", "pubmed_cc", "coauthorship_dblp", "cooking_200"]


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        raise ValueError("empty values")
    if len(vals) == 1:
        return vals[0], 0.0
    return st.mean(vals), st.stdev(vals)


def _fmt(ms: tuple[float, float]) -> str:
    return f"{ms[0]:+.2f} ± {ms[1]:.2f}"


def _load_probe(path: Path) -> dict:
    rows = list(csv.DictReader(path.open()))
    by_ds: dict[str, list[float]] = {}
    for row in rows:
        by_ds.setdefault(row["dataset"], []).append(float(row["delta_pp"]))
    ds_mean = {ds: sum(vals) / len(vals) for ds, vals in by_ds.items()}
    grand = sum(ds_mean.values()) / len(ds_mean)
    return {"rows": rows, "ds_mean": ds_mean, "grand": grand}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed13_dir", type=str, default="outputs_v2/s1_seed13_full_vs_nohca")
    ap.add_argument("--out_md", type=str, default="logs/s1_seed13_full_vs_nohca_summary.md")
    args = ap.parse_args()

    seed13_dir = ROOT / args.seed13_dir if not Path(args.seed13_dir).is_absolute() else Path(args.seed13_dir)
    out_md = ROOT / args.out_md if not Path(args.out_md).is_absolute() else Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    sources = {
        "full": {
            1: ROOT / "outputs_v2/p0_seed_sweep/seed_1/full/lodo_probe.csv",
            7: ROOT / "outputs_v2/p0_seed_sweep/seed_7/full/lodo_probe.csv",
            13: seed13_dir / "full/lodo_probe.csv",
        },
        "no_HCA": {
            1: ROOT / "outputs_v2/p0_seed_sweep/seed_1/no_HCA/lodo_probe.csv",
            7: ROOT / "outputs_v2/p0_seed_sweep/seed_7/no_HCA/lodo_probe.csv",
            13: seed13_dir / "no_HCA/lodo_probe.csv",
        },
    }
    loaded: dict[str, dict[int, dict]] = {}
    for variant, mapping in sources.items():
        loaded[variant] = {}
        for seed, path in mapping.items():
            if not path.is_file():
                raise FileNotFoundError(f"missing {variant} seed={seed}: {path}")
            loaded[variant][seed] = _load_probe(path)

    seed_wins: list[tuple[int, float, float]] = []
    for seed in (1, 7, 13):
        full_v = loaded["full"][seed]["grand"]
        no_hca_v = loaded["no_HCA"][seed]["grand"]
        seed_wins.append((seed, full_v, no_hca_v))

    no_hca_wins = sum(1 for _, full_v, no_hca_v in seed_wins if no_hca_v > full_v)
    full_wins = sum(1 for _, full_v, no_hca_v in seed_wins if full_v >= no_hca_v)

    full_stats = _mean_std([x[1] for x in seed_wins])
    no_hca_stats = _mean_std([x[2] for x in seed_wins])
    diff_stats = _mean_std([no_hca_v - full_v for _, full_v, no_hca_v in seed_wins])

    lines: list[str] = []
    lines.append("# S1-1 Full vs no_HCA (seed {1,7,13})")
    lines.append("")
    lines.append("| pretrain seed | full Δ (pp) | no_HCA Δ (pp) | no_HCA - full (pp) | winner |")
    lines.append("|---|---:|---:|---:|---|")
    for seed, full_v, no_hca_v in seed_wins:
        winner = "no_HCA" if no_hca_v > full_v else "full"
        lines.append(f"| {seed} | {full_v:+.2f} | {no_hca_v:+.2f} | {no_hca_v - full_v:+.2f} | {winner} |")

    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- full mean±std = {_fmt(full_stats)} pp")
    lines.append(f"- no_HCA mean±std = {_fmt(no_hca_stats)} pp")
    lines.append(f"- (no_HCA - full) mean±std = {_fmt(diff_stats)} pp")
    lines.append(f"- seed wins: no_HCA = {no_hca_wins}/3, full = {full_wins}/3")

    lines.append("")
    lines.append("## Criterion Verdict")
    lines.append("")
    if no_hca_wins >= 2:
        lines.append("- **Criterion triggered:** `no_HCA` wins in at least 2 of 3 seeds.")
        lines.append("- Suggested main configuration update: promote `CCA + HOR`, demote HCA from the core gain narrative.")
    else:
        lines.append("- **Criterion not triggered:** `full` is not beaten by `no_HCA` in a 2-of-3 majority.")
        lines.append("- Suggested action: keep `full` as the main configuration and write the seed1/7 anomaly as an observation.")

    lines.append("")
    lines.append("## Per-dataset mean±std across seeds")
    lines.append("")
    lines.append("| dataset | full Δ mean±std | no_HCA Δ mean±std | no_HCA - full mean±std |")
    lines.append("|---|---:|---:|---:|")
    for ds in DATASETS:
        full_vals = [loaded["full"][seed]["ds_mean"][ds] for seed in (1, 7, 13)]
        no_hca_vals = [loaded["no_HCA"][seed]["ds_mean"][ds] for seed in (1, 7, 13)]
        diff_vals = [b - a for a, b in zip(full_vals, no_hca_vals)]
        lines.append(
            f"| {ds} | {_fmt(_mean_std(full_vals))} | {_fmt(_mean_std(no_hca_vals))} | {_fmt(_mean_std(diff_vals))} |"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[summary] wrote {out_md}")


if __name__ == "__main__":
    main()
