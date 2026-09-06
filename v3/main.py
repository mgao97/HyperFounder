"""HyperGFSE pre-training entry point.

HyperGFSE follows the GFSE (WWW'26) paradigm: a structural encoder is trained
SELF-SUPERVISED (no labels) on a multi-domain pool of hypergraphs using four
tasks (HSPD regression, h-motif counting, hypergraph community detection, and
graph-level contrastive learning) with uncertainty-weighted loss. The frozen
encoder is later consumed by evaluate.py for downstream linear-probe evaluation.

Example (multi-domain pre-training):
    python main.py \
        --pretrain_datasets news20 ca_cora cc_cora cc_citeseer dblp4k_paper imdb_aw \
        --epochs 50 --output_dir outputs/hypergfse_pretrain --device cpu

For the full cross-domain transferability study, run scripts/run_lodo.sh which
holds out each dataset in turn (Leave-One-Dataset-Out cross-validation).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))

import numpy as np
import torch

from hypergfse.load_benchmark import load_benchmark, ALL_BENCHMARK_DATASETS
from hypergfse.encoder import HyperGFSE
from hypergfse.pretrain import Pretrainer, build_pretrain_item


def parse_args():
    p = argparse.ArgumentParser(description="HyperGFSE self-supervised pre-training")
    p.add_argument("--pretrain_datasets", nargs="+", required=True,
                   choices=ALL_BENCHMARK_DATASETS,
                   help="Datasets for self-supervised pre-training (multi-domain).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4, help="graphs per batch")
    p.add_argument("--max_nodes", type=int, default=3000,
                   help="Subsample graphs larger than this (SPD/community are O(N^2)).")
    p.add_argument("--pe_dim", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--output_dim", type=int, default=64)
    p.add_argument("--rw_mode", choices=["incidence", "clique"], default="incidence")
    p.add_argument("--size_invariant", action="store_true", default=True,
                   help="Damp large-hyperedge influence (our core contribution).")
    p.add_argument("--no_size_invariant", dest="size_invariant", action="store_false")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_pairs", type=int, default=20000,
                   help="Sampled node pairs for the pairwise SPD/community losses.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_dir", default="outputs/hypergfse_pretrain")
    return p.parse_args()


def make_domain_batch(items, domains, batch_size, rng):
    """Sample a batch that guarantees >=1 same-domain pair (GCL positives)."""
    n = len(items)
    anchor = int(rng.integers(n))
    ad = domains[anchor]
    siblings = [i for i in range(n) if domains[i] == ad and i != anchor]
    batch = [anchor]
    batch.append(int(rng.choice(siblings)) if siblings else int(rng.integers(n)))
    while len(batch) < batch_size:
        batch.append(int(rng.integers(n)))
    return [items[i] for i in batch]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # 1) load + pre-compute self-supervised labels (ONCE, they are static)
    items, domains = [], []
    print("Loading pre-training pool:")
    for di, name in enumerate(args.pretrain_datasets):
        g = load_benchmark(name, max_nodes=args.max_nodes, seed=args.seed)
        items.append(build_pretrain_item(g.H, ds_id=di, spd_d=args.pe_dim, motif_k=8, comm_thr=0.5))
        domains.append(di)
        s = g.raw_stats
        print(f"  {name:14s} domain={g.domain:12s} N={s['num_nodes']:6d} "
              f"E={s['num_hyperedges']:6d} avg|S|={s['avg_hyperedge_size']:6.1f} "
              f"classes={s['num_classes']}")

    # 2) model
    enc = HyperGFSE(pe_dim=args.pe_dim, hidden=args.hidden, num_layers=args.num_layers,
                    num_heads=args.num_heads, output_dim=args.output_dim, rw_mode=args.rw_mode,
                    size_invariant=args.size_invariant, beta=args.beta, dropout=args.dropout,
                    device=args.device)
    trainer = Pretrainer(enc, cfg={"motif_k": 8, "cd_eps": 1.0, "tau": 0.1,
                                   "lr": args.lr, "max_pairs": args.max_pairs})

    # 3) training loop
    n = len(items)
    print(f"\nPre-training {args.epochs} epochs on {n} graphs "
          f"(batch={args.batch_size}, max_nodes={args.max_nodes})...")
    for ep in range(args.epochs):
        epoch_loss, nb = 0.0, 0
        for _ in range(max(1, n // args.batch_size)):
            # make_domain_batch guarantees >=1 same-domain pair (GCL positives)
            batch = make_domain_batch(items, domains, args.batch_size, rng)
            per, tot = trainer.train_step(batch)
            epoch_loss += tot
            nb += 1
        print(f"  epoch {ep + 1:3d} | total {epoch_loss / max(1, nb):.4f}")

    # 4) save
    ckpt = {"config": vars(args), "encoder_state": enc.state_dict()}
    out_path = os.path.join(args.output_dir, "hypergfse_encoder.pt")
    torch.save(ckpt, out_path)
    with open(os.path.join(args.output_dir, "pretrain_meta.json"), "w") as f:
        json.dump({"datasets": args.pretrain_datasets,
                   "num_params": sum(p.numel() for p in enc.parameters())}, f, indent=2)
    print(f"Saved encoder checkpoint -> {out_path}")


if __name__ == "__main__":
    main()
