from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

from t_round_common import abs_path, checkpoint_fraction, checkpoint_loss, fmt_ms, grand_from_lodo_csv, mean_std, pearson_r

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ["none", "full"]
SEEDS = [1, 7, 13]
FRACTIONS = [0.25, 0.50, 0.75]


def frac_tag(frac: float) -> str:
    return f"{int(round(frac * 100)):02d}"


def load_points(base_dir: Path) -> list[dict]:
    points: list[dict] = []
    for seed in SEEDS:
        none_ckpts: dict[float, float] = {}
        for frac in FRACTIONS:
            ckpt = base_dir / "none" / f"seed_{seed}" / "checkpoints" / f"pretrain_frac_{frac_tag(frac)}_v2.pt"
            if ckpt.is_file():
                none_ckpts[frac] = checkpoint_loss(ckpt)
        for variant in VARIANTS:
            for frac in FRACTIONS:
                ckpt = base_dir / variant / f"seed_{seed}" / "checkpoints" / f"pretrain_frac_{frac_tag(frac)}_v2.pt"
                probe = base_dir / variant / f"seed_{seed}" / f"probe_frac_{frac_tag(frac)}.csv"
                if not ckpt.is_file() or not probe.is_file():
                    continue
                _, frozen = grand_from_lodo_csv(probe, "delta_pp")
                loss = checkpoint_loss(ckpt)
                none_loss = none_ckpts.get(frac)
                if none_loss is None:
                    continue
                points.append({
                    "variant": variant,
                    "seed": seed,
                    "fraction": frac,
                    "phase": "early" if frac <= 0.50 else "late",
                    "pretext_loss": loss,
                    "loss_reduction_vs_none": none_loss - loss,
                    "frozen_delta_pp": frozen,
                })
    return points


def corr_subset(points: list[dict]) -> float:
    xs = [float(p["loss_reduction_vs_none"]) for p in points]
    ys = [float(p["frozen_delta_pp"]) for p in points]
    return pearson_r(xs, ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="outputs_v2/t5_checkpoint_tradeoff")
    ap.add_argument("--out_md", type=str, default="logs/t5_checkpoint_tradeoff_summary.md")
    ap.add_argument("--out_dir", type=str, default="outputs_v2/figures")
    args = ap.parse_args()

    in_dir = abs_path(args.in_dir)
    out_md = abs_path(args.out_md)
    out_dir = abs_path(args.out_dir)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    points = load_points(in_dir)
    if not points:
        raise RuntimeError(f"no checkpoint/probe points found under {in_dir}")

    points_full = points
    points_early = [p for p in points if float(p["fraction"]) <= 0.50]
    points_late = [p for p in points if float(p["fraction"]) > 0.50]
    r_full = corr_subset(points_full)
    r_early = corr_subset(points_early)
    r_late = corr_subset(points_late)

    per_variant: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        subset = [p for p in points if p["variant"] == variant]
        per_variant[variant] = {
            "r_full": corr_subset(subset),
            "r_early": corr_subset([p for p in subset if float(p["fraction"]) <= 0.50]),
            "r_late": corr_subset([p for p in subset if float(p["fraction"]) > 0.50]),
        }

    pass_configs = []
    for variant, rs in per_variant.items():
        rf, re = rs["r_full"], rs["r_early"]
        if math.isfinite(rf) and math.isfinite(re) and abs(re) < abs(rf) - 0.2:
            pass_configs.append(variant)
    enough_configs = len(pass_configs) >= 2

    if math.isfinite(r_full) and math.isfinite(r_early) and abs(r_early) < abs(r_full) - 0.2 and enough_configs:
        verdict = "T5-PASS"
        verdict_msg = "trade-off 更像预训练后程过拟合驱动（动态机制）"
    elif math.isfinite(r_early) and math.isfinite(r_late) and abs(r_early - r_late) < 0.1:
        verdict = "T5-FAIL"
        verdict_msg = "trade-off 更像目标粒度错位驱动（目标机制）"
    else:
        verdict = "T5-INCONCLUSIVE"
        verdict_msg = "当前点集不足以严格满足预注册判据，需以 aggregate r 与 full 配置为主解释"

    lines = [
        "# T5 Checkpoint Trade-off Summary",
        "",
        f"- points = {len(points)}",
        f"- r_full = {r_full:+.3f}",
        f"- r_early (<=50%) = {r_early:+.3f}",
        f"- r_late (>50%) = {r_late:+.3f}",
        f"- criterion-satisfying configs = {pass_configs if pass_configs else '[]'}",
        f"- verdict = **{verdict}**: {verdict_msg}",
        "",
        "## Per-config Correlations",
        "",
        "| variant | r_full | r_early | r_late |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        rs = per_variant[variant]
        lines.append(f"| {variant} | {rs['r_full']:+.3f} | {rs['r_early']:+.3f} | {rs['r_late']:+.3f} |")

    lines += ["", "## Points", "", "| variant | seed | frac | loss reduction vs none | frozen Δ |", "|---|---:|---:|---:|---:|"]
    for p in sorted(points, key=lambda x: (x["variant"], x["seed"], x["fraction"])):
        lines.append(
            f"| {p['variant']} | {p['seed']} | {int(round(float(p['fraction']) * 100))}% | "
            f"{float(p['loss_reduction_vs_none']):+.4f} | {float(p['frozen_delta_pp']):+.2f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path = out_dir / "t5_checkpoint_tradeoff_points.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["variant", "seed", "fraction", "phase", "pretext_loss", "loss_reduction_vs_none", "frozen_delta_pp"],
        )
        w.writeheader()
        w.writerows(points)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.0, 6.0), constrained_layout=True)
    frac_colors = {0.25: "#4c78a8", 0.50: "#f58518", 0.75: "#54a24b"}
    marker_map = {"none": "s", "full": "o"}
    for p in points:
        ax.scatter(
            float(p["loss_reduction_vs_none"]),
            float(p["frozen_delta_pp"]),
            s=95,
            marker=marker_map[str(p["variant"])],
            color=frac_colors[float(p["fraction"])],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
        )
    xs = [float(p["loss_reduction_vs_none"]) for p in points if p["variant"] == "full"]
    ys = [float(p["frozen_delta_pp"]) for p in points if p["variant"] == "full"]
    if len(xs) >= 2 and len(set(xs)) > 1:
        m, b = (0.0, sum(ys) / len(ys))
        try:
            import numpy as np
            m, b = np.polyfit(xs, ys, deg=1)
        except Exception:
            pass
        x0, x1 = min(xs), max(xs)
        ax.plot([x0, x1], [m * x0 + b, m * x1 + b], linestyle="--", color="#7f8c8d", linewidth=1.5)

    ax.set_xlabel("Pretext loss reduction vs none", fontsize=12)
    ax.set_ylabel("Frozen probe Δ (pp)", fontsize=12)
    ax.set_title("T5 Checkpoint Trade-off (25/50/75%)", fontsize=14, fontweight="bold")
    ax.text(0.02, 0.98, f"r_full={r_full:+.2f}\nr_early={r_early:+.2f}\nr_late={r_late:+.2f}",
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#bbbbbb"),
            fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    png_path = out_dir / "t5_checkpoint_tradeoff_scatter.png"
    pdf_path = out_dir / "t5_checkpoint_tradeoff_scatter.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("\n".join(lines))
    print(f"\n[summary] wrote {out_md}")
    print(f"[plot] wrote {png_path}")
    print(f"[plot] wrote {pdf_path}")
    print(f"[points] wrote {csv_path}")


if __name__ == "__main__":
    main()
