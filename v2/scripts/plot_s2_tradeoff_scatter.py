from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROWS = ["none", "CCA", "HCA", "HOR", "CCA_HCA", "CCA_HOR", "HCA_HOR", "full"]


def _pretext_best_from_log(path: Path) -> float:
    txt = path.read_text(encoding="utf-8")
    matches = re.findall(r"best ckpt epoch=\d+ loss=([0-9.]+)", txt)
    if not matches:
        raise RuntimeError(f"no best loss found in {path}")
    return float(matches[-1])


def _grand_from_csv(path: Path, key: str) -> float:
    rows = list(csv.DictReader(path.open()))
    by_ds: dict[str, list[float]] = {}
    for row in rows:
        by_ds.setdefault(row["dataset"], []).append(float(row[key]))
    ds_mean = {ds: sum(vals) / len(vals) for ds, vals in by_ds.items()}
    return sum(ds_mean.values()) / len(ds_mean)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="outputs_v2/figures")
    ap.add_argument("--scratch_x", type=float, default=None, help="Optional x-position for the scratch star.")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points: list[dict[str, float | str]] = []
    for row in ROWS:
        pretext = _pretext_best_from_log(ROOT / f"logs/p1_grid_train_{row}.log")
        frozen = _grand_from_csv(ROOT / f"outputs_v2/p1_tradeoff_grid/{row}/frozen.csv", "delta_pp")
        finetune = _grand_from_csv(ROOT / f"outputs_v2/p1_tradeoff_grid/{row}/finetune.csv", "delta_vs_raw_pp")
        points.append({"row": row, "pretext_best_loss": pretext, "frozen_delta_pp": frozen, "finetune_delta_pp": finetune})

    scratch = _grand_from_csv(ROOT / "outputs_v2/p0_frozen_vs_finetune/frozen_scratch.csv", "delta_pp")
    none_loss = next(float(p["pretext_best_loss"]) for p in points if p["row"] == "none")
    xs = np.array([none_loss - float(p["pretext_best_loss"]) for p in points], dtype=float)
    ys = np.array([float(p["frozen_delta_pp"]) for p in points], dtype=float)
    slope, intercept = np.polyfit(xs, ys, deg=1)
    fit_x = np.linspace(xs.min() - 0.001, xs.max() + 0.001, 128)
    fit_y = slope * fit_x + intercept
    corr = float(np.corrcoef(xs, ys)[0, 1])
    scratch_x = args.scratch_x if args.scratch_x is not None else (float(xs.min()) - 0.0018)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 6.1), constrained_layout=True)

    palette = {
        "grid": "#2f4259",
        "scratch": "#cc4631",
        "fit": "#7f8c8d",
    }

    for point, x in zip(points, xs):
        row = str(point["row"])
        y = float(point["frozen_delta_pp"])
        size = 120 if row != "full" else 130
        ax.scatter(
            x,
            y,
            s=size,
            marker="o",
            color=palette["grid"],
            edgecolor=palette["grid"],
            linewidth=0.6,
            zorder=3,
        )
        dx = 0.00045
        dy = 0.06
        if row == "HOR":
            dy = 0.12
        elif row == "CCA_HOR":
            dy = 0.06
        elif row == "full":
            dy = 0.06
        elif row == "CCA_HCA":
            dy = 0.04
        elif row == "HCA_HOR":
            dy = 0.06
        elif row == "HCA":
            dy = 0.06
        ax.text(x + dx, y + dy, row, fontsize=10)

    ax.scatter(
        scratch_x,
        scratch,
        s=340,
        marker="*",
        color=palette["scratch"],
        edgecolor=palette["scratch"],
        linewidth=1.0,
        zorder=4,
        label="scratch (no pretext)",
    )
    ax.text(scratch_x + 0.00045, scratch - 0.16, "scratch", fontsize=10, color=palette["scratch"])

    ax.plot(fit_x, fit_y, color=palette["fit"], linestyle="--", linewidth=1.5, zorder=2, label=f"linear fit (grid only, r = {corr:+.2f})")

    grid_handle = ax.scatter([], [], s=120, marker="o", color=palette["grid"], label="module grid")
    scratch_handle = ax.scatter([], [], s=340, marker="*", color=palette["scratch"], label="scratch (no pretext)")
    fit_handle = ax.plot([], [], color=palette["fit"], linestyle="--", linewidth=1.5, label=f"linear fit (grid only, r = {corr:+.2f})")[0]

    ax.set_xlabel("Pretext loss reduction vs none (↓ loss − none)", fontsize=13)
    ax.set_ylabel("Frozen probe Δ (pp)", fontsize=13)
    ax.grid(True, alpha=0.22)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=[grid_handle, scratch_handle, fit_handle], loc="lower right", frameon=False, fontsize=10)

    ax.set_xlim(min(xs.min(), scratch_x) - 0.0015, xs.max() + 0.0025)
    ax.set_ylim(min(ys.min(), scratch) - 0.6, max(ys.max(), scratch) + 0.5)

    png_path = out_dir / "pretext_vs_probe_scatter.png"
    pdf_path = out_dir / "pretext_vs_probe_scatter.pdf"
    csv_path = out_dir / "pretext_vs_probe_scatter_points.csv"
    legacy_png_path = out_dir / "s2_tradeoff_pretext_vs_frozen.png"
    legacy_pdf_path = out_dir / "s2_tradeoff_pretext_vs_frozen.pdf"
    legacy_csv_path = out_dir / "s2_tradeoff_pretext_vs_frozen_points.csv"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(legacy_png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(legacy_pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["row", "pretext_best_loss", "pretext_loss_reduction_vs_none", "frozen_delta_pp", "finetune_delta_pp"],
        )
        w.writeheader()
        for point, x in zip(points, xs):
            w.writerow(
                {
                    "row": point["row"],
                    "pretext_best_loss": point["pretext_best_loss"],
                    "pretext_loss_reduction_vs_none": x,
                    "frozen_delta_pp": point["frozen_delta_pp"],
                    "finetune_delta_pp": point["finetune_delta_pp"],
                }
            )
    legacy_csv_path.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"[plot] wrote {png_path}")
    print(f"[plot] wrote {pdf_path}")
    print(f"[plot] wrote {csv_path}")
    print(f"[plot] wrote {legacy_png_path}")
    print(f"[plot] wrote {legacy_pdf_path}")
    print(f"[plot] wrote {legacy_csv_path}")
    print(f"[plot] pearson_r={corr:+.4f} slope={slope:+.4f} intercept={intercept:+.4f}")
    print(f"[plot] scratch_star_x={scratch_x:+.4f} scratch_frozen_delta={scratch:+.4f} (excluded from fit)")


if __name__ == "__main__":
    main()
