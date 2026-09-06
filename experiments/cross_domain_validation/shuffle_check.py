#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Standalone P0 shuffle ablation.

Imports run_cdv as a module (no main() run) and, in a SINGLE process so the
loaded datasets are shared, computes:
  M0      = fixed structure + no encoding
  M2      = fixed structure + H-LDP
  M2_shuf = fixed structure + H-LDP with node rows independently permuted
The comparison M2_shuf vs {M0, M2} answers whether the *arrangement* of H-LDP
across nodes carries transferable signal (a real structural encoding) or only
its marginal distribution matters.
"""
import sys
import csv
import numpy as np

sys.path.insert(0, ".")
import run_cdv as R  # noqa: E402

rows = []
print("=== P0 shuffle ablation (single process, internally consistent) ===")
for pair_id, src_name, tgt_name, dom_s, dom_t, note in R.PAIRS:
    src = R.load_dataset(src_name)
    tgt = R.load_dataset(tgt_name)
    s_e, s_fixed = R.fixed_structure(src)
    t_e, t_fixed = R.fixed_structure(tgt)
    print(f"\n--- {pair_id} ({src_name} -> {tgt_name}) ---")
    for seed in R.SEEDS:
        nf0_s = R.enc_none(s_fixed)
        nf0_t = R.enc_none(t_fixed)
        m0 = R.transfer_auroc(src, tgt, nf0_s, nf0_t, seed)
        nf2_s = R.enc_hldp(s_e, src["n_nodes"])
        nf2_t = R.enc_hldp(t_e, tgt["n_nodes"])
        m2 = R.transfer_auroc(src, tgt, nf2_s, nf2_t, seed)
        m2s = R.transfer_auroc_hldp_shuffle(src, tgt, seed)
        print(f"  seed={seed}  M0={m0:.4f}  M2={m2:.4f}  M2_shuf={m2s:.4f}")
        rows.append(dict(pair=pair_id, seed=seed,
                         m0=round(m0, 4), m2=round(m2, 4), m2_shuf=round(m2s, 4)))

print("\n=== means over 3 seeds ===")
out = []
for pair_id, src_name, tgt_name, dom_s, dom_t, note in R.PAIRS:
    sub = [r for r in rows if r["pair"] == pair_id]
    m0 = float(np.mean([r["m0"] for r in sub]))
    m2 = float(np.mean([r["m2"] for r in sub]))
    m2s = float(np.mean([r["m2_shuf"] for r in sub]))
    line = (f"{pair_id}: M0={m0:.4f}  M2={m2:.4f}  M2_shuf={m2s:.4f}  "
            f"| M2-M0={m2-m0:+.4f}  M2_shuf-M0={m2s-m0:+.4f}")
    print(line)
    out.append(line)

with open("results/shuffle_check.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["pair", "seed", "m0", "m2", "m2_shuf"])
    w.writeheader()
    w.writerows(rows)
print("\n[written] results/shuffle_check.csv")
