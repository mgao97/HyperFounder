from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from t_round_common import (
    abs_path,
    best_pretext_loss_from_best_ckpt,
    fmt_ms,
    grand_from_lodo_csv,
    mean_std,
    pearson_r,
    pooled_std,
)

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [1, 7, 13]
BETAS = ["1e-4", "1e-3", "1e-2"]


def load_seed_points(base_dir: Path, group: str, none_ref_dir: Path) -> list[dict]:
    points: list[dict] = []
    for seed in SEEDS:
        best_ckpt = base_dir / group / f"seed_{seed}" / "checkpoints" / "pretrain_best_v2.pt"
        probe_csv = base_dir / group / f"seed_{seed}" / "frozen.csv"
        none_ckpt = none_ref_dir / "none" / f"seed_{seed}" / "checkpoints" / "pretrain_best_v2.pt"
        if not best_ckpt.is_file() or not probe_csv.is_file() or not none_ckpt.is_file():
            continue
        _, frozen = grand_from_lodo_csv(probe_csv, "delta_pp")
        loss = best_pretext_loss_from_best_ckpt(best_ckpt)
        none_loss = best_pretext_loss_from_best_ckpt(none_ckpt)
        points.append({
            "seed": seed,
            "pretext_best_loss": loss,
            "loss_reduction_vs_none": none_loss - loss,
            "frozen_delta_pp": frozen,
        })
    return points


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="outputs_v2/t1_ib")
    ap.add_argument("--none_ref_dir", type=str, default="outputs_v2/t5_checkpoint_tradeoff")
    ap.add_argument("--out_md", type=str, default="logs/t1_ib_summary.md")
    ap.add_argument("--out_dir", type=str, default="outputs_v2/figures")
    args = ap.parse_args()

    in_dir = abs_path(args.in_dir)
    none_ref_dir = abs_path(args.none_ref_dir)
    out_md = abs_path(args.out_md)
    out_dir = abs_path(args.out_dir)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    g2 = load_seed_points(in_dir, "g2_cca_hor_baseline", none_ref_dir)
    if not g2:
        raise RuntimeError("missing G2 baseline points")
    g2_vals = [float(p["frozen_delta_pp"]) for p in g2]
    g2_stats = mean_std(g2_vals)
    g2_r = pearson_r([float(p["loss_reduction_vs_none"]) for p in g2], g2_vals)

    g1_by_beta = {beta: load_seed_points(in_dir, f"g1_none_ib_b{beta}", none_ref_dir) for beta in BETAS}
    g3_by_beta = {beta: load_seed_points(in_dir, f"g3_cca_hor_ib_b{beta}", none_ref_dir) for beta in BETAS}

    lines = [
        "# T1 IB Summary",
        "",
        f"- G2 baseline mean±std = {fmt_ms(g2_stats)} pp",
        f"- G2 trade-off r = {g2_r:+.3f}",
        "",
        "## G1 (none + IB) scan",
        "",
        "| beta | frozen Δ mean±std | trade-off r |",
        "|---|---:|---:|",
    ]
    for beta in BETAS:
        pts = g1_by_beta[beta]
        if not pts:
            lines.append(f"| {beta} | missing | missing |")
            continue
        vals = [float(p["frozen_delta_pp"]) for p in pts]
        r = pearson_r([float(p["loss_reduction_vs_none"]) for p in pts], vals)
        lines.append(f"| {beta} | {fmt_ms(mean_std(vals))} | {r:+.3f} |")

    lines += ["", "## G3 (CCA+HOR + IB) scan", "", "| beta | frozen Δ mean±std | trade-off r | (a) beat G2 by 2×pooled std | (b) |r| reduce by 0.2 | verdict |", "|---|---:|---:|---|---|---|"]
    best_beta = None
    best_rank = -1
    verdict_rank = {"T1-FAIL": 0, "T1-MODERATE": 1, "T1-STRONG": 2}
    plot_rows: list[dict] = []
    for beta in BETAS:
        pts = g3_by_beta[beta]
        if not pts:
            lines.append(f"| {beta} | missing | missing | missing | missing | T1-FAIL |")
            continue
        vals = [float(p["frozen_delta_pp"]) for p in pts]
        stats = mean_std(vals)
        r = pearson_r([float(p["loss_reduction_vs_none"]) for p in pts], vals)
        cond_a = stats[0] >= g2_stats[0] + 2.0 * pooled_std(vals, g2_vals)
        cond_b = abs(r) <= abs(g2_r) - 0.2
        if cond_a and cond_b:
            verdict = "T1-STRONG"
        elif cond_a:
            verdict = "T1-MODERATE"
        else:
            verdict = "T1-FAIL"
        if verdict_rank[verdict] > best_rank:
            best_rank = verdict_rank[verdict]
            best_beta = beta
        lines.append(
            f"| {beta} | {fmt_ms(stats)} | {r:+.3f} | {'YES' if cond_a else 'NO'} | {'YES' if cond_b else 'NO'} | {verdict} |"
        )
        for p in pts:
            plot_rows.append({"group": f"G3 beta={beta}", **p})

    lines += ["", f"## Recommendation", "", f"- best_beta = {best_beta}", f"- best_verdict = {list(verdict_rank.keys())[best_rank] if best_rank >= 0 else 'T1-FAIL'}"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # plot
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.0, 6.0), constrained_layout=True)
    color_map = {"G2": "#2f4259", "1e-4": "#4c78a8", "1e-3": "#f58518", "1e-2": "#54a24b"}
    for p in g2:
        ax.scatter(float(p["loss_reduction_vs_none"]), float(p["frozen_delta_pp"]), s=110, color=color_map["G2"], edgecolor="black", linewidth=0.5)
    for beta in BETAS:
        for p in g3_by_beta[beta]:
            ax.scatter(float(p["loss_reduction_vs_none"]), float(p["frozen_delta_pp"]), s=95, marker="^", color=color_map[beta], edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Pretext loss reduction vs none", fontsize=12)
    ax.set_ylabel("Frozen probe Δ (pp)", fontsize=12)
    ax.set_title("T1 G2 vs G3 IB Trade-off", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    png_path = out_dir / "t1_ib_tradeoff_overlay.png"
    pdf_path = out_dir / "t1_ib_tradeoff_overlay.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    csv_path = out_dir / "t1_ib_points.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "seed", "pretext_best_loss", "loss_reduction_vs_none", "frozen_delta_pp"])
        w.writeheader()
        for p in g2:
            w.writerow({"group": "G2", **p})
        for beta in BETAS:
            for p in g3_by_beta[beta]:
                w.writerow({"group": f"G3_beta_{beta}", **p})

    print("\n".join(lines))
    print(f"\n[summary] wrote {out_md}")
    print(f"[plot] wrote {png_path}")
    print(f"[plot] wrote {pdf_path}")
    print(f"[points] wrote {csv_path}")


if __name__ == "__main__":
    main()
