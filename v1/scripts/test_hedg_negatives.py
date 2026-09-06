"""
Smoke test for HEDG-Weighted Hard Negative Sampling.

Loads the real DHG datasets (cora_cc, cooking_200, etc.) and exercises
the HEDG sampler end-to-end:
  1. Build the HEDG for each dataset and report stats.
  2. Sample hard negatives for a batch of positive edges.
  3. Report the average HEDG similarity of sampled negatives.
  4. Compare against random sampling baseline (HEDG = uniform).
  5. Compare against a temperature sweep (τ = 0.1, 0.5, 1.0, 2.0).

This script is meant to be quick (<1 minute on CPU) and serves as
a sanity check that the HEDG approach works on real hypergraph data
before plugging it into a full pretraining run.

Usage:
    python scripts/test_hedg_negatives.py
    python scripts/test_hedg_negatives.py --datasets cora_cc cooking_200
    python scripts/test_hedg_negatives.py --temperature 0.5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from v1.models.hedg_negative_sampling import HEDGNegativeSampler
from utils.common import set_seed
from utils.dhg_datasets import load_dhg_sample


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test for HEDG negative sampler.")
    p.add_argument("--datasets", nargs="+", default=["cora_cc", "cooking_200"],
                   help="Datasets to test on (must be in DHG and cacheable).")
    p.add_argument("--temperature", type=float, default=0.5,
                   help="HEDG sampling temperature τ.")
    p.add_argument("--num-negatives", type=int, default=2)
    p.add_argument("--num-samples", type=int, default=50,
                   help="Number of positive edges to sample from per dataset.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--compare-uniform", action="store_true", default=True,
                   help="Also run uniform (τ=∞) sampling as a baseline.")
    return p.parse_args()


def _summarize_sampler(sampler: HEDGNegativeSampler, dataset_name: str,
                       num_pos: int, generator) -> dict:
    """Sample positives from a uniform random subset, then summarize."""
    pos_edges = torch.randint(0, sampler.num_edges, (num_pos,), generator=generator).tolist()
    t0 = time.perf_counter()
    result = sampler.sample_hyperedge_negatives(pos_edges, generator=generator)
    elapsed = time.perf_counter() - t0
    if result.neg_similarities.numel() == 0:
        return {
            "dataset": dataset_name,
            "num_pos_used": 0,
            "num_neg": 0,
            "avg_hedg_sim": 0.0,
            "max_hedg_sim": 0.0,
            "fallback_rate": 0.0,
            "elapsed_sec": elapsed,
        }
    sims = result.neg_similarities.tolist()
    return {
        "dataset": dataset_name,
        "num_pos_used": int(result.pos_edge_indices.numel()),
        "num_neg": int(result.neg_similarities.numel()),
        "avg_hedg_sim": sum(sims) / len(sims),
        "max_hedg_sim": max(sims),
        "min_hedg_sim": min(sims),
        "fallback_rate": result.meta.get("n_random_fallback", 0) / max(
            result.meta.get("n_hard_used", 0) + result.meta.get("n_random_fallback", 0), 1
        ),
        "elapsed_sec": elapsed,
    }


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    print("=" * 76)
    print("HEDG-Weighted Hard Negative Sampling — Smoke Test")
    print(f"Datasets: {args.datasets}")
    print(f"Temperature: τ = {args.temperature}")
    print(f"Num negatives per positive: {args.num_negatives}")
    print(f"Num positive edges sampled per dataset: {args.num_samples}")
    print("=" * 76)

    for dataset_name in args.datasets:
        print(f"\n--- {dataset_name} ---")
        try:
            hg = load_dhg_sample(dataset_name, target_dim=64, seed=args.seed,
                                 data_root=str(PROJECT_ROOT / "data" / "cache"))
        except Exception as e:
            print(f"  [skip] failed to load {dataset_name}: {e}")
            continue

        # HEDG stats
        t0 = time.perf_counter()
        sampler = HEDGNegativeSampler(
            hg,
            temperature=args.temperature,
            num_negatives=args.num_negatives,
            seed=args.seed,
        )
        build_time = time.perf_counter() - t0
        stats = sampler.get_hedg_stats()
        print(f"  HEDG build time: {build_time*1000:.1f} ms")
        print(f"  HEDG stats: edges={stats['num_edges']}, "
              f"hedg_edges={stats['num_hedg_edges']}, "
              f"avg_overlap={stats['avg_overlap']:.2f}, "
              f"max_overlap={stats['max_overlap']:.0f}, "
              f"density={stats['hedg_density']:.3f}")

        # Main sample with default temperature
        s = _summarize_sampler(sampler, dataset_name, args.num_samples, generator)
        print(f"  τ = {args.temperature:>4.2f}  |  "
              f"num_pos={s['num_pos_used']:>3d}  num_neg={s['num_neg']:>3d}  "
              f"avg_sim={s['avg_hedg_sim']:.2f}  max_sim={s['max_hedg_sim']:.0f}  "
              f"fallback={s['fallback_rate']:.2f}  "
              f"time={s['elapsed_sec']*1000:.1f} ms")

        # Compare against uniform (τ=∞) sampling as baseline
        if args.compare_uniform:
            print(f"  --- temperature sweep ---")
            for tau in [0.1, 0.5, 1.0, 5.0, 100.0]:
                sampler_t = HEDGNegativeSampler(
                    hg, temperature=tau, num_negatives=args.num_negatives, seed=args.seed
                )
                s = _summarize_sampler(sampler_t, dataset_name, args.num_samples, generator)
                print(f"  τ = {tau:>6.2f}  |  "
                      f"avg_sim={s['avg_hedg_sim']:.2f}  max_sim={s['max_hedg_sim']:.0f}  "
                      f"fallback={s['fallback_rate']:.2f}")

    print()
    print("=" * 76)
    print("Interpretation:")
    print("  - 'avg_hedg_sim' = average HEDG similarity (shared-node count) of")
    print("    the sampled negatives. HIGHER = harder negatives.")
    print("  - 'max_hedg_sim' = max HEDG similarity seen. Higher means the")
    print("    sampler found at least one very similar neighbor.")
    print("  - 'fallback' = fraction of positives that fell back to random")
    print("    (no HEDG neighbor with overlap >= threshold).")
    print("  - As τ decreases, avg_sim should INCREASE (sharper focus on")
    print("    top-K most similar HEDG neighbors).")
    print("=" * 76)


if __name__ == "__main__":
    main()
