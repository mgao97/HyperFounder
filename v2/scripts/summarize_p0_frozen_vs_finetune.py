from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_csv(path: Path):
    rows = list(csv.DictReader(path.open()))
    by_ds = {}
    metric_key = "ours_acc" if "ours_acc" in rows[0] else "test_acc"
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(float(r[metric_key]))
    ds_mean = {k: sum(v) / len(v) for k, v in by_ds.items()}
    grand = sum(ds_mean.values()) / len(ds_mean)
    return rows, ds_mean, grand, metric_key


def main() -> None:
    files = {
        "frozen_full": ROOT / "outputs_v2/p0_frozen_vs_finetune/frozen_full.csv",
        "frozen_scratch": ROOT / "outputs_v2/p0_frozen_vs_finetune/frozen_scratch.csv",
        "finetune_full": ROOT / "outputs_v2/p0_frozen_vs_finetune/finetune_full.csv",
        "finetune_scratch": ROOT / "outputs_v2/p0_frozen_vs_finetune/finetune_scratch.csv",
    }
    loaded = {k: load_csv(v) for k, v in files.items()}

    lines = ["# P0-2 frozen vs finetune summary", "",
             "| protocol | scratch mean acc% | pretrained mean acc% | gap (pp) |",
             "|---|---:|---:|---:|"]
    for protocol in ("frozen", "finetune"):
        s = loaded[f"{protocol}_scratch"][2] * 100.0
        f = loaded[f"{protocol}_full"][2] * 100.0
        lines.append(f"| {protocol} | {s:.2f} | {f:.2f} | {f - s:+.2f} |")

    lines += ["", "## Discussion cue", ""]
    f_gap = loaded["frozen_full"][2] * 100.0 - loaded["frozen_scratch"][2] * 100.0
    t_gap = loaded["finetune_full"][2] * 100.0 - loaded["finetune_scratch"][2] * 100.0
    lines.append(f"- frozen gap = {f_gap:+.2f} pp")
    lines.append(f"- finetune gap = {t_gap:+.2f} pp")
    if t_gap > 0:
        lines.append("- finetune retains a positive pretraining gain.")
    else:
        lines.append("- finetune removes or reverses the pretraining gain; discussion should scope claims to frozen transfer.")

    target = ROOT / "logs" / "p0_frozen_vs_finetune_summary.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[summary] wrote {target}")


if __name__ == "__main__":
    main()
