from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROWS = ["none", "CCA", "HCA", "HOR", "CCA_HCA", "CCA_HOR", "HCA_HOR", "full"]


def grand_from_csv(path: Path, key: str) -> float:
    rows = list(csv.DictReader(path.open()))
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(float(r[key]))
    ds_mean = {k: sum(v) / len(v) for k, v in by_ds.items()}
    return sum(ds_mean.values()) / len(ds_mean)


def pretext_best_from_log(path: Path) -> float:
    txt = path.read_text(encoding="utf-8")
    matches = re.findall(r"best ckpt epoch=\d+ loss=([0-9.]+)", txt)
    if not matches:
        raise RuntimeError(f"no best loss found in {path}")
    return float(matches[-1])


def main() -> None:
    lines = ["# P1-3 trade-off grid summary", "",
             "| row | pretext best loss | frozen Δ (pp) | finetune Δ vs raw (pp) |",
             "|---|---:|---:|---:|"]
    none_pre = None
    none_frozen = None
    none_finetune = None
    tmp = {}
    for row in ROWS:
        pre = pretext_best_from_log(ROOT / f"logs/p1_grid_train_{row}.log")
        frozen = grand_from_csv(ROOT / f"outputs_v2/p1_tradeoff_grid/{row}/frozen.csv", "delta_pp")
        finetune = grand_from_csv(ROOT / f"outputs_v2/p1_tradeoff_grid/{row}/finetune.csv", "delta_vs_raw_pp")
        tmp[row] = (pre, frozen, finetune)
        if row == "none":
            none_pre, none_frozen, none_finetune = pre, frozen, finetune
        lines.append(f"| {row} | {pre:.4f} | {frozen:+.2f} | {finetune:+.2f} |")

    lines += ["", "## Trade-off check vs none", "",
              "| row | pretext improvement vs none | frozen Δ shift vs none | finetune Δ shift vs none |",
              "|---|---:|---:|---:|"]
    for row in ROWS:
        if row == "none":
            continue
        pre, frozen, finetune = tmp[row]
        lines.append(
            f"| {row} | {none_pre - pre:+.4f} | {frozen - none_frozen:+.2f} | {finetune - none_finetune:+.2f} |"
        )

    target = ROOT / "logs" / "p1_tradeoff_grid_summary.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[summary] wrote {target}")


if __name__ == "__main__":
    main()
