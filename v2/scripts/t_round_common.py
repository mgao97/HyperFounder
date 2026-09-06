from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["cora_cc", "citeseer_cc", "pubmed_cc", "coauthorship_dblp", "cooking_200"]


def abs_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def mean_std(vals: Iterable[float]) -> tuple[float, float]:
    xs = list(float(v) for v in vals)
    if not xs:
        raise ValueError("empty values")
    if len(xs) == 1:
        return xs[0], 0.0
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return mean, math.sqrt(max(var, 0.0))


def pooled_std(a: Iterable[float], b: Iterable[float]) -> float:
    a_s = mean_std(a)[1]
    b_s = mean_std(b)[1]
    return math.sqrt(a_s ** 2 + b_s ** 2)


def fmt_ms(ms: tuple[float, float], signed: bool = True) -> str:
    mean, std = ms
    return f"{mean:+.2f} ± {std:.2f}" if signed else f"{mean:.2f} ± {std:.2f}"


def grand_from_lodo_csv(path: str | Path, key: str = "delta_pp") -> tuple[dict[str, float], float]:
    csv_path = abs_path(path)
    rows = list(csv.DictReader(csv_path.open()))
    by_ds: dict[str, list[float]] = {}
    for row in rows:
        by_ds.setdefault(row["dataset"], []).append(float(row[key]))
    ds_mean = {ds: sum(vals) / len(vals) for ds, vals in by_ds.items()}
    grand = sum(ds_mean.values()) / len(ds_mean)
    return ds_mean, grand


def checkpoint_loss(path: str | Path) -> float:
    ckpt = torch.load(abs_path(path), map_location="cpu")
    if "loss" in ckpt:
        return float(ckpt["loss"])
    if "best_loss" in ckpt:
        return float(ckpt["best_loss"])
    raise KeyError(f"checkpoint has no loss field: {path}")


def checkpoint_fraction(path: str | Path) -> float | None:
    ckpt = torch.load(abs_path(path), map_location="cpu")
    frac = ckpt.get("fraction")
    return None if frac is None else float(frac)


def checkpoint_epoch(path: str | Path) -> int | None:
    ckpt = torch.load(abs_path(path), map_location="cpu")
    epoch = ckpt.get("epoch")
    return None if epoch is None else int(epoch)


def pretext_best_from_log(path: str | Path) -> float:
    txt = abs_path(path).read_text(encoding="utf-8")
    matches = re.findall(r"best ckpt epoch=\d+ loss=([0-9.]+)", txt)
    if not matches:
        raise RuntimeError(f"no best loss found in {path}")
    return float(matches[-1])


def best_pretext_loss_from_best_ckpt(path: str | Path) -> float:
    ckpt = torch.load(abs_path(path), map_location="cpu")
    val = ckpt.get("best_pretext_loss")
    if val is not None:
        return float(val)
    if ckpt.get("best_loss") is not None:
        return float(ckpt["best_loss"])
    raise KeyError(f"checkpoint has no best_pretext_loss/best_loss: {path}")


def pearson_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs)
    den_y = sum((y - my) ** 2 for y in ys)
    if den_x <= 0 or den_y <= 0:
        return float("nan")
    return num / math.sqrt(den_x * den_y)
