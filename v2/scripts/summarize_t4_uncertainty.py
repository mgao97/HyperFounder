from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from t_round_common import abs_path, fmt_ms, grand_from_lodo_csv, mean_std, pooled_std

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [1, 7, 13]


def load_group_points(base_dir: Path, group: str) -> list[dict]:
    pts: list[dict] = []
    for seed in SEEDS:
        probe_csv = base_dir / group / f"seed_{seed}" / "frozen.csv"
        epoch_csv = base_dir / group / f"seed_{seed}" / "logs" / "pretrain_epochs_v2.csv"
        if not probe_csv.is_file() or not epoch_csv.is_file():
            continue
        _, frozen = grand_from_lodo_csv(probe_csv, "delta_pp")
        rows = list(csv.DictReader(epoch_csv.open()))
        last = rows[-1]
        logvs = [float(last["uw_logv_mlm"]), float(last["uw_logv_mem"]), float(last["uw_logv_dual"])]
        pts.append({
            "seed": seed,
            "frozen_delta_pp": frozen,
            "uw_logv_mlm": logvs[0],
            "uw_logv_mem": logvs[1],
            "uw_logv_dual": logvs[2],
            "high_gap": max(logvs) - sorted(logvs)[1],
            "epoch_csv": str(epoch_csv),
        })
    return pts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="outputs_v2/t4_uncertainty")
    ap.add_argument("--baseline_dir", type=str, default="outputs_v2/t1_ib")
    ap.add_argument("--out_md", type=str, default="logs/t4_uncertainty_summary.md")
    ap.add_argument("--out_dir", type=str, default="outputs_v2/figures")
    args = ap.parse_args()

    in_dir = abs_path(args.in_dir)
    baseline_dir = abs_path(args.baseline_dir)
    out_md = abs_path(args.out_md)
    out_dir = abs_path(args.out_dir)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_group_points(baseline_dir, "g2_cca_hor_baseline")
    h1 = load_group_points(in_dir, "h1_full_residual_uw")
    h2 = load_group_points(in_dir, "h2_cca_hor_residual_uw")
    if not baseline:
        raise RuntimeError("missing baseline G2 points for T4 comparison")

    base_vals = [float(p["frozen_delta_pp"]) for p in baseline]
    base_stats = mean_std(base_vals)
    h1_vals = [float(p["frozen_delta_pp"]) for p in h1]
    h2_vals = [float(p["frozen_delta_pp"]) for p in h2]
    h1_stats = mean_std(h1_vals) if h1_vals else (float("nan"), float("nan"))
    h2_stats = mean_std(h2_vals) if h2_vals else (float("nan"), float("nan"))

    h1_gain = bool(h1_vals) and h1_stats[0] >= base_stats[0] + 2.0 * pooled_std(h1_vals, base_vals)
    h2_gain = bool(h2_vals) and h2_stats[0] >= base_stats[0] + 2.0 * pooled_std(h2_vals, base_vals)
    # Proxy assumption: since T4 implementation weights pretext branches, "auto-detected instability"
    # means one branch's final log-variance stands out by > 0.2 in >=2 seeds.
    h1_auto_detect = sum(1 for p in h1 if float(p["high_gap"]) > 0.2) >= 2

    if h1_gain and h1_auto_detect:
        verdict = "T4-STRONG"
    elif h2_gain:
        verdict = "T4-MODERATE"
    else:
        verdict = "T4-FAIL"

    lines = [
        "# T4 Uncertainty Summary",
        "",
        f"- baseline G2 mean±std = {fmt_ms(base_stats)} pp",
        f"- H1 mean±std = {fmt_ms(h1_stats)} pp" if h1_vals else "- H1 missing",
        f"- H2 mean±std = {fmt_ms(h2_stats)} pp" if h2_vals else "- H2 missing",
        f"- H1 beat baseline by 2×pooled std = {'YES' if h1_gain else 'NO'}",
        f"- H2 beat baseline by 2×pooled std = {'YES' if h2_gain else 'NO'}",
        f"- H1 auto-detected high-uncertainty branch in >=2 seeds = {'YES' if h1_auto_detect else 'NO'}",
        f"- verdict = **{verdict}**",
        "",
        "> Assumption: T4 code follows the implementation paragraph of the plan, i.e. task-level",
        "> log-variances `s_mlm/s_mem/s_dual` are tracked. The summary therefore uses a branch-gap proxy",
        "> instead of a literal `s_HCA`, which is not a directly defined variable in the current pretext setup.",
        "",
        "## Per-seed Final Log-Variances",
        "",
        "| group | seed | frozen Δ | logv_mlm | logv_mem | logv_dual | max-gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group_name, pts in [("H1", h1), ("H2", h2)]:
        for p in pts:
            lines.append(
                f"| {group_name} | {p['seed']} | {float(p['frozen_delta_pp']):+.2f} | "
                f"{float(p['uw_logv_mlm']):+.3f} | {float(p['uw_logv_mem']):+.3f} | "
                f"{float(p['uw_logv_dual']):+.3f} | {float(p['high_gap']):+.3f} |"
            )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True, sharey=True)
    for ax, group_name, pts in zip(axes, ["H1", "H2"], [h1, h2]):
        for p in pts:
            rows = list(csv.DictReader(Path(p["epoch_csv"]).open()))
            epochs = [int(r["epoch"]) + 1 for r in rows]
            ax.plot(epochs, [float(r["uw_logv_mlm"]) for r in rows], label=f"seed{p['seed']}-mlm")
            ax.plot(epochs, [float(r["uw_logv_mem"]) for r in rows], linestyle="--", label=f"seed{p['seed']}-mem")
            ax.plot(epochs, [float(r["uw_logv_dual"]) for r in rows], linestyle=":", label=f"seed{p['seed']}-dual")
        ax.set_title(group_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("log-variance")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    png_path = out_dir / "t4_uncertainty_logv_trajectories.png"
    pdf_path = out_dir / "t4_uncertainty_logv_trajectories.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("\n".join(lines))
    print(f"\n[summary] wrote {out_md}")
    print(f"[plot] wrote {png_path}")
    print(f"[plot] wrote {pdf_path}")


if __name__ == "__main__":
    main()
