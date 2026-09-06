import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs_v2" / "u_round" / "u_round_results.csv"
FIG_DIR = ROOT / "outputs_v2" / "figures"
LOG_DIR = ROOT / "logs"
OUT_MD = LOG_DIR / "u2_subspace_removal_summary.md"


def _mean_std(xs):
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def _pooled_std(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt(a[1] ** 2 + b[1] ** 2)


def _load_rows(p: Path):
    with open(p, "r", newline="") as f:
        return list(csv.DictReader(f))


def _grand_delta_stats(rows, acc_key):
    """Return dataset-level grand mean and pooled-like std as (mean, std)."""
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(float(r[acc_key]) - float(r["baseline_acc"]))
    dataset_means = [float(np.mean(v)) for v in by_ds.values() if v]
    return _mean_std(dataset_means)


def main():
    rows = _load_rows(SRC)
    rows = [r for r in rows if r["dataset"] != "gowalla"]

    # Aggregate per (group, seed, frac) so that checkpoints are comparable
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["group"], int(r["seed"]), int(r["fraction"]))].append(r)

    checkpoint_rows = []
    for (g, s, f), rs in grouped.items():
        z_m, z_s = _grand_delta_stats(rs, "Z_acc")
        row = {"group": g, "seed": s, "frac": f,
               "Z_delta": z_m, "Z_delta_std": z_s}
        for variant in ["ov", "pca", "rand"]:
            for k in [1, 2, 3]:
                key = f"{variant}{k}_acc"
                m, sd = _grand_delta_stats(rs, key)
                row[f"{variant}{k}_delta"] = m
                row[f"{variant}{k}_delta_std"] = sd
        checkpoint_rows.append(row)

    # Main verdict: U2-C:
    #   S_ov removal delta >= Z_delta + 2*pooled_std
    #   AND S_ov removal is strictly better than both controls (rand & top-k PCA)
    best_k_rows = []
    for cr in checkpoint_rows:
        base = (cr["Z_delta"], cr["Z_delta_std"])
        best_k = None
        for k in [1, 2, 3]:
            ov = (cr[f"ov{k}_delta"], cr[f"ov{k}_delta_std"])
            rand = (cr[f"rand{k}_delta"], cr[f"rand{k}_delta_std"])
            pca = (cr[f"pca{k}_delta"], cr[f"pca{k}_delta_std"])
            margin_vs_z = ov[0] - base[0]
            thr_vs_z = 2.0 * _pooled_std(ov, base)
            strong_cond = (margin_vs_z > thr_vs_z) and (ov[0] > rand[0]) and (ov[0] > pca[0])
            moderate_cond = (ov[0] > base[0]) and (ov[0] > rand[0])
            if best_k is None:
                best_k = (k, margin_vs_z, thr_vs_z, strong_cond, moderate_cond, ov[0], rand[0], pca[0])
            else:
                if margin_vs_z > best_k[1]:
                    best_k = (k, margin_vs_z, thr_vs_z, strong_cond, moderate_cond, ov[0], rand[0], pca[0])
        cr["best_k"] = best_k[0]
        cr["best_margin"] = best_k[1]
        cr["best_thr"] = best_k[2]
        cr["strong"] = best_k[3]
        cr["moderate"] = best_k[4]
        cr["best_ov_delta"] = best_k[5]
        cr["best_rand_delta"] = best_k[6]
        cr["best_pca_delta"] = best_k[7]
        best_k_rows.append(cr)

    # checkpoint-level verdict ratios
    n = len(best_k_rows)
    strong_cnt = sum(1 for c in best_k_rows if c["strong"])
    moderate_cnt = sum(1 for c in best_k_rows if (c["moderate"] and not c["strong"]))
    fail_cnt = n - strong_cnt - moderate_cnt

    # overall verdict based on cross-checkpoint majority / signal strength
    margins = np.array([c["best_margin"] for c in best_k_rows], dtype=float)
    m_mean, m_std = _mean_std(margins.tolist())
    best_ovs = np.array([c["best_ov_delta"] for c in best_k_rows], dtype=float)
    best_pcas = np.array([c["best_pca_delta"] for c in best_k_rows], dtype=float)
    best_rands = np.array([c["best_rand_delta"] for c in best_k_rows], dtype=float)
    ov_vs_pca = (best_ovs - best_pcas).mean()
    ov_vs_rand = (best_ovs - best_rands).mean()

    overall = "U2-FAIL"
    if strong_cnt >= max(1, n // 3) and ov_vs_pca > 0 and ov_vs_rand > 0:
        overall = "U2-STRONG"
    elif moderate_cnt + strong_cnt >= max(1, n // 3):
        overall = "U2-MODERATE"

    # Write markdown
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# U2：过拟合子空间识别与切除 — 汇总判据")
    lines.append("")
    lines.append(f"- 结果 CSV：[u_round_results.csv]({SRC.as_posix()})")
    lines.append(f"- Checkpoint 级聚合行数：{n}")
    lines.append("")
    lines.append("## 总体判定口径")
    lines.append("")
    lines.append("- **STRONG**：切除 S_ov 后 Δ ≥ 原表征 + 2×pooled_std，且严格优于随机方向 / 谱 top-k 两个对照。")
    lines.append("- **MODERATE**：切除有效但不显著优于 PCA top-k（或仅部分条件满足）。")
    lines.append("- **FAIL**：切除无增益或 S_ov 不承载可分离的过拟合子空间。")
    lines.append("")
    lines.append("## checkpoint-level 命中")
    lines.append("")
    lines.append(f"| 指标 | 值 |")
    lines.append("|---|---:|")
    lines.append(f"| total checkpoints | {n} |")
    lines.append(f"| U2-STRONG 命中 | {strong_cnt} |")
    lines.append(f"| U2-MODERATE 命中（不含 STRONG） | {moderate_cnt} |")
    lines.append(f"| U2-FAIL 命中 | {fail_cnt} |")
    lines.append(f"| best S_ov margin vs Z（均值） | {m_mean:+.4f} pp |")
    lines.append(f"| best S_ov margin vs Z（std） | {m_std:.4f} |")
    lines.append(f"| best S_ov Δ − best PCA Δ（均值） | {ov_vs_pca:+.4f} pp |")
    lines.append(f"| best S_ov Δ − best rand Δ（均值） | {ov_vs_rand:+.4f} pp |")
    lines.append(f"| 总体判定 | **{overall}** |")
    lines.append("")

    lines.append("## 每 checkpoint 的最佳维度对照")
    lines.append("")
    lines.append("| group | seed | frac | Z Δ (pp) | best k | S_ov Δ (pp) | PCA Δ (pp) | rand Δ (pp) | margin-vs-Z | 2σ thr | strong | moderate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for cr in sorted(best_k_rows, key=lambda c: (c["group"], c["seed"], c["frac"])):
        k = cr["best_k"]
        lines.append(
            f"| {cr['group']} | {cr['seed']} | {cr['frac']} | {cr['Z_delta']:+.2f} | {k} | {cr['best_ov_delta']:+.2f} | {cr['best_pca_delta']:+.2f} | {cr['best_rand_delta']:+.2f} | {cr['best_margin']:+.3f} | {cr['best_thr']:.3f} | {'YES' if cr['strong'] else ''} | {'YES' if cr['moderate'] else ''} |")
    lines.append("")

    # Figure: grouped bar per checkpoint, best-k Δ between Z / S_ov / PCA / rand
    # To keep readable, average across seeds (plot group × frac × variant)
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.0), sharey=True)
    for ax, group in zip(axes, ["none", "cca_hor"]):
        sub = [c for c in best_k_rows if c["group"] == group]
        if not sub:
            continue
        sub = sorted(sub, key=lambda c: (c["frac"], c["seed"]))
        labels = [f"{c['frac']}%·s{c['seed']}" for c in sub]
        xs = np.arange(len(sub))
        width = 0.2
        base = [c["Z_delta"] for c in sub]
        ov = [c["best_ov_delta"] for c in sub]
        pca = [c["best_pca_delta"] for c in sub]
        rand = [c["best_rand_delta"] for c in sub]
        ax.bar(xs - 1.5 * width, base, width=width, label="Z (frozen)", color="#4C78A8")
        ax.bar(xs - 0.5 * width, ov, width=width, label="Z − P_{S_ov}Z", color="#E45756")
        ax.bar(xs + 0.5 * width, pca, width=width, label="Z − PCA_topk", color="#72B7B2")
        ax.bar(xs + 1.5 * width, rand, width=width, label="Z − Random", color="#BAB0AC")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("grand Δ (pp)")
        ax.set_title(f"{group}：子空间切除前后 frozen probe Δ（每 checkpoint 取最佳 k）")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    p_bar = FIG_DIR / "u2_subspace_removal_grouped_bars.pdf"
    fig.savefig(p_bar, dpi=220, bbox_inches="tight")
    plt.close(fig)

    lines.append("## 图输出")
    lines.append("")
    lines.append(f"- 子空间切除 Δ 对照柱状图：[u2_subspace_removal_grouped_bars.pdf]({p_bar.as_posix()})")
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[u2] wrote {OUT_MD}")
    print(f"[u2] wrote {p_bar}")


if __name__ == "__main__":
    main()
