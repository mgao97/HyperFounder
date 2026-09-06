"""Downstream linear-probe evaluation of a frozen HyperGFSE encoder.

Protocol (GFSE WWW'26 "pre-train -> linear-probe"):
  1. Load a frozen HyperGFSE encoder from a checkpoint.
  2. For each eval dataset: compute PSE = enc(H) (no grad) and concatenate
     [raw_features | PSE] per node.
  3. Stratified K-fold: train a LogisticRegression on the train fold, report
     mean/std test accuracy. Also report the raw-features-only baseline.
  4. Optionally evaluate a random-init encoder as an ablation.

This isolates the QUALITY OF THE STRUCTURAL ENCODING: the encoder is NOT
finetuned, so any gain over the raw-features baseline comes purely from the
pretrained structure signal.

Examples:
    # evaluate a cross-domain checkpoint on all 8 datasets
    python evaluate.py --checkpoint outputs/hypergfse_pretrain/hypergfse_encoder.pt \
        --eval_datasets all --folds 10 --output outputs/hypergfse_eval.csv

    # single held-out dataset
    python evaluate.py --checkpoint outputs/lodo/fold_0/hypergfse_encoder.pt \
        --eval_datasets news20 --random_baseline
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from hypergfse.load_benchmark import load_benchmark, ALL_BENCHMARK_DATASETS
from hypergfse.encoder import HyperGFSE


def _probe(X: np.ndarray, y: np.ndarray, folds: int, seed: int):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs)), float(np.std(accs))


def evaluate_encoder(enc, name, max_nodes, folds, seed, device) -> dict:
    g = load_benchmark(name, max_nodes=max_nodes, seed=seed)
    x = g.x.numpy().astype(np.float32)
    y = g.y
    with torch.no_grad():
        pse = enc.forward(g.H).cpu().numpy().astype(np.float32)  # (N, output_dim)
    X_cat = np.concatenate([x, pse], axis=1)
    acc_raw, std_raw = _probe(x, y, folds, seed)
    acc_pse, std_pse = _probe(X_cat, y, folds, seed)
    return {
        "dataset": name, "domain": g.domain,
        "nodes": g.raw_stats["num_nodes"], "classes": g.num_classes,
        "raw_acc": acc_raw, "raw_std": std_raw,
        "pse_acc": acc_pse, "pse_std": std_pse,
        "delta": acc_pse - acc_raw,
    }


def _build_encoder(cfg, device):
    return HyperGFSE(pe_dim=cfg["pe_dim"], hidden=cfg["hidden"], num_layers=cfg["num_layers"],
                     num_heads=cfg["num_heads"], output_dim=cfg["output_dim"], rw_mode=cfg["rw_mode"],
                     size_invariant=cfg["size_invariant"], beta=cfg["beta"], dropout=cfg["dropout"],
                     device=device)


def main():
    p = argparse.ArgumentParser(description="HyperGFSE downstream linear-probe evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to hypergfse_encoder.pt")
    p.add_argument("--eval_datasets", nargs="+", default=["all"],
                   choices=ALL_BENCHMARK_DATASETS + ["all"])
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--max_nodes", type=int, default=None,
                   help="Sub-sampling size; defaults to the value used at pre-train time.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="outputs/hypergfse_eval.csv")
    p.add_argument("--random_baseline", action="store_true",
                   help="Also evaluate a randomly-initialised encoder (no pre-train).")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = ckpt["config"]
    max_nodes = args.max_nodes if args.max_nodes is not None else cfg.get("max_nodes")

    enc = _build_encoder(cfg, args.device)
    enc.load_state_dict(ckpt["encoder_state"])
    enc.eval()
    print(f"Loaded encoder from {args.checkpoint} (max_nodes={max_nodes})")

    datasets = ALL_BENCHMARK_DATASETS if "all" in args.eval_datasets else args.eval_datasets
    rows = []
    for name in datasets:
        r = evaluate_encoder(enc, name, max_nodes, args.folds, args.seed, args.device)
        print(f"  {name:14s} raw={r['raw_acc']:.4f}±{r['raw_std']:.3f}  "
              f"+PSE={r['pse_acc']:.4f}±{r['pse_std']:.3f}  Δ={r['delta']:+.4f}")
        rows.append(r)

    if args.random_baseline:
        enc_r = _build_encoder(cfg, args.device)
        enc_r.eval()
        for name in datasets:
            r = evaluate_encoder(enc_r, name, max_nodes, args.folds, args.seed + 1, args.device)
            r = {**r, "dataset": name + " (random-init)"}
            print(f"  {r['dataset']:24s} +PSE={r['pse_acc']:.4f}±{r['pse_std']:.3f}")
            rows.append(r)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "domain", "nodes", "classes",
                                          "raw_acc", "raw_std", "pse_acc", "pse_std", "delta"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
