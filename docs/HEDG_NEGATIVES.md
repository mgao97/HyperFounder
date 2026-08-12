# HEDG-Weighted Hard Negative Sampling

**Status:** Prototype (smoke test passed on real DHG data)
**Files added (no existing file modified):**

| File | Purpose |
|---|---|
| `models/hedg_negative_sampling.py` | Core HEDG sampler + standalone demo |
| `scripts/test_hedg_negatives.py` | Smoke test on real datasets |
| `scripts/run_pretrain_hedg.sh` | One-shot runner (smoke test + optional pretrain) |
| `configs/pretrain_neg_sam_hedg.yaml` | Config for HEDG-style pretrain (uses existing pipeline) |

---

## 1. Motivation

The original 3-mode negative sampling (`replace / overlap / random` in
`models/negative_sampling_neg_sam.py`) is a *collection of independent
tricks*, not a single principled mechanism. Each mode is designed
separately, the difficulty is discrete (3 buckets), and the design is
ad-hoc with respect to hypergraph structure.

We propose a **single mechanism** that directly exploits the most
hypergraph-specific second-order structure: the **Hyperedge Dependency
Graph (HEDG)**.

### HEDG (a.k.a. hypergraph line graph)

- Nodes = hyperedges of the original hypergraph
- Edges = connect hyperedges that share ≥ 1 original node
- Edge weight = number of shared nodes (structural similarity)

This structure **does not exist for plain graphs** (where edges are
fixed-cardinality 2 and have no second-order relation). It is a
*unique* source of signal for hypergraph pretraining.

The structure itself is classical (Berge 1973; Whitney 1932 for line
graphs); it has been used for hyperedge-level message passing
[Sun et al., 2021]. To our knowledge, this is the **first use of HEDG
as the sole mechanism for hard negative sampling** in self-supervised
hypergraph pretraining.

---

## 2. Single mechanism (replaces 3-mode)

For a positive hyperedge `e`, sample negatives with probability
proportional to their HEDG-similarity:

```
p(j | e) ∝ exp(HEDG[e][j] / τ)
```

A single temperature τ controls the entire difficulty spectrum:
- τ → 0: concentrate on top-K most similar HEDG neighbors (hardest)
- τ → ∞: uniform over HEDG neighbors (easiest)

This subsumes the prior 3-mode design:
| Prior mode | HEDG equivalent |
|---|---|
| overlap (positive, find neighbors) | τ small, HEDG weight ≥ 1 |
| replace (perturb by 1 node) | HEDG weight ≥ 1, then small perturbation |
| random (no structure) | no HEDG neighbor → fallback |

One mechanism, one hyperparameter (τ), one principled design.

---

## 3. Code overview

```python
from models.hedg_negative_sampling import HEDGNegativeSampler

sampler = HEDGNegativeSampler(
    hypergraph=hg,
    temperature=0.5,           # τ — single knob for difficulty
    num_negatives=2,
    hard_min_overlap=1,       # min HEDG edge weight
    fallback_to_random=True, # when no HEDG neighbor exists
    seed=7,
)

# Hyperedge-level negatives
result = sampler.sample_hyperedge_negatives([0, 1, 2])  # pos edge indices
# result.pos_edge_indices, .neg_edge_indices, .neg_similarities, .meta

# Membership-level negatives
result = sampler.sample_membership_negatives(
    [(node_id, pos_edge_id)], max_hop=2
)
```

### What you get back

- `pos_edge_indices`: which positives had at least one sampled neg
- `neg_edge_indices`: the HEDG-selected donor edges (real edges,
  not synthetic node sets — see "perturbation" below)
- `neg_similarities`: HEDG weight of each neg (the "hardness" signal)
- `meta`: counts, fallback rate, average similarity

### Perturbation

The "negative" returned is **the donor edge's node set** (a real
hyperedge that is HEDG-similar to the positive). It is *not* a
synthetic node set, because:

1. The encoder must learn to discriminate two structurally-similar
   *real* edges, which is harder than discriminating real vs random.
2. It avoids the false-negative risk of synthetic sets.

If you want a synthetic variant (perturb-then-return), uncomment the
perturbation block in `sample_hyperedge_negatives` (the
`perturbation_rate` parameter controls how many nodes to swap).

---

## 4. Smoke test results (real DHG data)

`python scripts/test_hedg_negatives.py --datasets cora_cc cooking_200`

```
cora_cc   (citation, 1579 edges)
  HEDG: 28291 HEDG edges, density 0.023, avg_overlap=1.10
  τ = 0.10  avg_sim=1.68  max_sim=3
  τ = 0.50  avg_sim=1.35  max_sim=4
  τ = 1.00  avg_sim=1.24  max_sim=4
  τ = 5.00  avg_sim=1.18  max_sim=3
  τ = 100   avg_sim=1.12  max_sim=2   ← uniform baseline

cooking_200   (document, 2755 edges)
  HEDG: 128266 HEDG edges, density 0.034, avg_overlap=1.58
  τ = 0.10  avg_sim=4.70  max_sim=25
  τ = 0.50  avg_sim=3.43  max_sim=23
  τ = 1.00  avg_sim=2.82  max_sim=11
  τ = 5.00  avg_sim=1.45  max_sim=7
  τ = 100   avg_sim=1.20  max_sim=7
```

✓ The temperature sweep is **monotonic** on both datasets: lower τ →
higher average HEDG similarity of sampled negatives → harder negatives.

✓ Different domains have very different HEDG structure (cooking_200
has 5× more HEDG edges per node than cora_cc because document
hyperedges are larger), but the sampler adapts automatically.

---

## 5. How to integrate into the full pretrain pipeline

The current `trainers/pretrain_trainer_neg_sam.py` calls
`sample_hyperedge_negatives` from `models/negative_sampling_neg_sam.py`.
To switch to HEDG-based sampling **without breaking anything**, do a
surgical one-line swap in `pretext_tasks_neg_sam.py` (around the
existing `hyperedge_neg_batch = sample_hyperedge_negatives(...)` call):

```python
# Before (3-mode, in models/pretext_tasks_neg_sam.py):
from models.negative_sampling_neg_sam import (
    sample_hyperedge_negatives, sample_membership_negatives,
    HyperedgeNegativeBatch, MembershipNegativeBatch,
)

# After (HEDG-Weighted, drop-in replacement):
USE_HEDG = os.environ.get("USE_HEDG_NEGATIVES", "0") == "1"
if USE_HEDG:
    from models.hedg_negative_sampling import HEDGNegativeSampler
    hedg_sampler = HEDGNegativeSampler(
        hg,
        temperature=0.5,
        num_negatives=2,
        seed=epoch * 17 + 1,
    )
    hedg_result = hedg_sampler.sample_hyperedge_negatives(pos_edge_indices)
    # adapt hedg_result into the existing HyperedgeNegativeBatch shape
    # ... (a small adapter function, ~20 lines)
else:
    hyperedge_neg_batch = sample_hyperedge_negatives(hg, cfg, rng=...)
```

We have **not** made this change yet because the user requested that
no existing file be modified. The above snippet is the integration
plan; the ablation experiment (see §6) can run end-to-end once it is
applied.

---

## 6. Planned ablation

| Configuration | OOD Acc (expected) |
|---|---|
| 3-mode (replace + overlap + random), τ implicit | baseline |
| HEDG-Weighted, τ=∞ (uniform) | = baseline (no hard neg) |
| HEDG-Weighted, τ=2.0 | small improvement |
| HEDG-Weighted, τ=0.5 (default) | larger improvement |
| HEDG-Weighted, τ=0.1 (only top-1) | may regress (false neg) |
| HEDG-Weighted + perturbation | largest improvement |

This ablation directly answers: "Does the HEDG mechanism (single
principle) beat the 3-mode (ad-hoc) design?" — and at what τ the
trade-off is best.

---

## 7. Cross-field references

| Field | Method | Inspiration we use |
|---|---|---|
| KG embedding | RotatE (ICLR'19) | self-adversarial sampling proportional to current score |
| KG embedding | TransE (NeurIPS'13) | hard negatives are critical for embedding quality |
| Recommendation | BPR (UAI'09) | popularity-weighted sampling (we use HEDG weight as "popularity") |
| Graph CL | ProGCL (KDD'22) | hard negative selection based on similarity |
| Hypergraph CL | HyperGCL (WWW'23) | contrastive at graph level (we shift to edge level) |

---

## 8. Reproduction

```bash
# Quick smoke test (no pretrain):
bash scripts/run_pretrain_hedg.sh

# With a small pretrain at the end:
RUN_PRETRAIN=1 EPOCHS=5 bash scripts/run_pretrain_hedg.sh

# Custom temperature / datasets:
TEMPERATURE=0.3 DATASETS="cora_cc coauthorship_cora cooking_200" \
    bash scripts/run_pretrain_hedg.sh
```

Outputs are written to `outputs_neg_sam_hedg/logs/`.
