from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from utils.common import ensure_dir, save_json
from utils.metrics import summarize_scores


def write_loss_history(path: str, history: List[Dict[str, float]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    if not history:
        save_json(target.with_suffix(".json"), {"history": []})
        return
    fieldnames = sorted(history[0].keys())
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def write_ablation_csv(path: str, rows: List[Dict[str, float | str]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    fieldnames = sorted(rows[0].keys()) if rows else ["task", "score"]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_seed_runs(scores: List[float], metric_name: str) -> Dict[str, float]:
    summary = summarize_scores(scores)
    return {metric_name: summary["mean"], f"{metric_name}_std": summary["std"]}
