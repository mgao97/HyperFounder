# Experimental Design — `neg_sam_v2` Pretraining

**Last updated:** 2026-08-12
**Status:** Draft for review
**Reference frameworks:** GFSE (WWW'26) for protocol, Hyper-FM (TPAMI'25) for cross-domain motivation, IHP (KDD'24) for link-prediction eval design

---

## 0. Design Philosophy

We follow the GFSE evaluation framework:
- **Multi-domain pretraining** (multiple datasets, multiple domains) for transferable structural features.
- **Standard splits** for downstream (60/20/20 random; OGB official for OGB-style datasets).
- **Explicit in-domain / OOD split** for downstream — avoid leakage.
- **Mean ± std over 5 seeds** for robustness.
- **Two complementary eval protocols**: linear probe (clean signal) + full finetune (realistic).

Three changes vs GFSE, motivated by Hyper-FM and IHP:
- **Linear probe** added on top of full finetune (cleaner signal of encoder quality).
- **Three downstream task types** (node classification, hyperedge prediction, link prediction) instead of one.
- **Multi-task self-supervised pretraining** (8 tasks) instead of 4, justified by the lack of pretrained encoders on hypergraph structural features.

---

## 1. Pretraining Stage

### 1.1 Datasets (8 datasets × 4 domains)

Aligned with GFSE's "more domains ≠ just more data" finding (Hyper-FM scaling law §V-D).

| Domain | Datasets | Nodes | Edges | Avg edge size | Why include |
|---|---|---:|---:|---:|---|
| **citation** (×3) | cora_cc | 2,708 | 1,579 | 3.03 | classic cocitation benchmark |
| | citeseer_cc | 3,312 | 1,079 | 3.20 | smaller cocitation, multi-domain robustness |
| | pubmed_cc | 19,717 | 7,963 | 2.50 | largest citation, tests scale |
| **academic** (×2) | coauthorship_cora | 2,708 | 1,072 | 4.28 | author-level co-occurrence |
| | coauthorship_dblp | 42,564 | 89,790 | 2.10 | large academic, dense bipartite |
| **document** (×2) | cooking_200 | 7,403 | 2,755 | 19.96 | ingredient recipe, dense |
| | news20 | 15,962 | 624 | 25.60 | document-word, very dense |
| **recommendation** (×2) | gowalla | 40,182 | 1,027,370 | 25.60 | large bipartite social check-in |
| | yelp_2018 | 91,270 | 1,237,439 | 13.60 | largest, user-item review |

**Why 4 domains, not 5–6:**
- Each domain contributes ≥ 2 datasets (covers the "per-domain multi-instance" principle GFSE uses).
- The 4 domains span **structurally different** hypergraph types (citation: small hyperedges; document: large hyperedges; recommendation: large bipartite).
- Going to 5–6 domains (political, commerce) is reserved for **OOD testing only**, not pretraining — this is the *critical* discipline.

**Why 8 datasets, not more:**
- GFSE used 8, Hyper-FM used 11 but did NOT separate OOD; we need to leave OOD space.
- Total node count: ~225k — comparable to GFSE (~4.6M total) considering we focus on structure not text.
- Computational budget: 300 epochs × 32 steps = ~10k steps, fits in 1 GPU × 3–4 hours.

### 1.2 Pretrain-train split

```yaml
# Pretraining does NOT need train/val/test split.
# All nodes are used; the encoder never sees downstream labels.
splits: {}  # or train_ratio: 1.0
```

Validation is monitored **per step** (every `log_interval_steps` epochs), based on training loss, NOT a held-out split.

### 1.3 Pretraining Tasks (8 self-supervised tasks)

We prune from the original 13 to **8 tasks** (justified below). All run jointly with task-specific uncertainty weighting (or fixed weights; ablation in §1.5).

#### Group A — Node-level representation (2 tasks)

| # | Task | Loss | What it teaches |
|---|---|---|---|
| 1 | **Masked Node Prediction** (MNP) | MSE on masked features | Local feature reconstruction; "what should this node look like?" |
| 2 | **Domain Alignment** (DA, GRL) | Cross-entropy on domain id (gradient-reversed) | Cross-domain robustness; encoder learns domain-invariant features |

#### Group B — Hyperedge-level representation (2 tasks)

| # | Task | Loss | What it teaches |
|---|---|---|---|
| 3 | **Hyperedge Discrimination** (HR, neg_sam core) | log-sigmoid margin on pos/neg edge score | Hyperedge boundary detection; "is this set of nodes a real edge?" |
| 4 | **Membership Contrast** (MC, neg_sam core) | log-sigmoid margin on node-edge membership | Node-edge interaction; "does this node belong to this edge?" — uses hop-based hard negatives |

#### Group C — View-invariant representation (1 task)

| # | Task | Loss | What it teaches |
|---|---|---|---|
| 5 | **Cross-View Contrastive** (CL) | InfoNCE on two augmented views | View invariance; "same node, different dropouts, same embedding" |

#### Group D — Cross-domain disentanglement (3 tasks)

| # | Task | Loss | What it teaches |
|---|---|---|---|
| 6 | **Orthogonality Loss** (Orth) | 1 − cos²(shared, private) | Disentanglement; shared and private branches carry non-redundant info |
| 7 | **Private Domain Prediction** (PDP) | Cross-entropy on domain id (private branch) | Domain-specific preservation; private branch keeps domain info |
| 8 | **Multi-Granularity Alignment** (MGA) | InfoNCE on shared↔domain prototypes | Cross-domain prototype anchoring |

#### Why these 8 (and not the other 5 we dropped)

| Dropped | Reason |
|---|---|
| ~~Hyperedge Size Prediction~~ | Too simple (almost any encoder learns cardinality); low information density |
| ~~Motif Classification~~ | Expensive to compute; motif distributions vary widely across domains, poor transferability |
| ~~Community Prototype Alignment~~ | Overlaps with MGA at the prototype-alignment level |
| ~~Structure Alignment~~ | Cross-view CL (#5) already enforces view invariance at the node level; graph-level cosine adds little |
| ~~Structure Discrimination~~ | Already covered by quality-aware routing (§1.4) which filters weak samples |

### 1.4 Quality-aware routing (auxiliary mechanism, not a "task")

Already implemented in `utils/minibatch_sampling.py::compute_subhypergraph_quality`. Each sampled subhypergraph gets a quality score; tasks gate themselves by threshold. No additional pretraining loss — it's a **sampling** mechanism.

### 1.5 Loss weighting

Two options:
- **(Default) Fixed weights** (from `pretrain_neg_sam_v2.yaml::loss_weights`).
- **(Optional, ablation) Task uncertainty weighting** (Kendall et al.) — automatically balances 8 tasks.

Default weights (sum ≈ 4.0 to keep magnitudes comparable):

| Task | Weight |
|---|---|
| MNP | 1.0 |
| HR | 1.0 |
| MC | 0.5 |
| CL | 1.0 |
| DA | 0.1 |
| Orth-node | 0.02 |
| Orth-edge | 0.02 |
| PDP-node | 0.05 |
| PDP-edge | 0.05 |
| MGA | 0.2 |

### 1.6 Pretraining hyperparameters

| Param | Value | Source |
|---|---|---|
| Optimizer | AdamW (lr=1e-3, wd=1e-4) | GFSE default |
| Batch size | 32 steps × ~4 subgraphs | existing config |
| Epochs | 300 (early stop patience 80) | existing |
| bf16 autocast | yes | added by our fix |
| Gradient clip | 1.0 | added by our fix |
| Seeds | 1 (seed=7); reproducibility | standard |

---

## 2. Downstream Stage

### 2.1 Downstream tasks (3 types)

We add a third task type (hyperedge prediction) beyond GFSE/Hyper-FM/IHP.

#### Task 1 — Node Classification (primary)

**Goal:** classify each node into one of $C$ classes.

**Datasets:**

| Set | Dataset | Domain | Notes |
|---|---|---|---|
| **In-domain** | cora | citation | standard split (DHG standard) |
| | citeseer | citation | standard split |
| | pubmed | citation | standard split |
| | coauthorship_cora | academic | in pretrain via `coauthorship_cora`, fine-tune via `coauthorship_cora` |
| | cooking_200 | document | in pretrain |
| **OOD** | house_committees | political | NOT in pretrain; tests new domain |
| | walmart_trips | commerce | NOT in pretrain; tests new domain |
| | coauthorship_dblp (only if we DON'T pretrain on it) | academic | optional held-out |

⚠️ **Decision needed**: do we pretrain on `coauthorship_dblp` (large, valuable) or hold it out for OOD?
- **Recommendation:** pretrain on it (large-scale data matters for pretraining), then test OOD on `house_committees` + `walmart_trips` + `coauthorship_cora`.

**Split:** 60 / 20 / 20 random (or Planetoid 60/20/20 for cora/citeseer/pubmed).

**Metrics:** Accuracy, Macro-F1.

#### Task 2 — Hyperedge Prediction (new, hypergraph-specific)

**Goal:** given a partial hyperedge (some nodes observed, some hidden), predict the missing nodes.

**Setup:**
1. Take a hypergraph and randomly hide $\rho_h$ fraction of node memberships within each hyperedge (typical: $\rho_h = 0.3$).
2. Encoder produces node embeddings $h_v$.
3. For each query (partial hyperedge $e_q$ with $|e_q|$ observed), rank all candidate nodes by score $s(v, e_q) = h_v^\top W h_{e_q}$.
4. Report Recall@K and MRR.

**Datasets:**
- Same as Node Classification (in-domain + OOD split applies).
- For recommendation datasets: this is equivalent to "predict item for user" → the same as link prediction.

**Metrics:** Recall@10, Recall@20, MRR.

**Why include:** hyperedge prediction is the **defining task** of hypergraph learning (impossible for plain GNN); if our pretraining helps on this, it is a clean signal of hypergraph-specific structural knowledge.

#### Task 3 — Link Prediction (for recommendation datasets)

**Goal:** predict held-out user-item interactions (treat hyperedges as bipartite).

**Datasets:** gowalla, yelp_2018 (in pretrain), movielens_1m (OOD).

**Setup:**
- Bipartite user-item graph; 80/10/10 random split on edges.
- Negative sampling: 99 random items per positive.

**Metrics:** Recall@10, Recall@20, NDCG@10, NDCG@20.

**Source:** Inspired by IHP's link-prediction focus.

### 2.2 Downstream splits

| Task | Split rule |
|---|---|
| Node classification | 60 / 20 / 20 random, seed=7 (Planetoid for cora/citeseer/pubmed) |
| Hyperedge prediction | 60 / 20 / 20 split on edges (train edges observed, val/test edges masked); per-node features from full graph |
| Link prediction | 80 / 10 / 10 random on edges |

### 2.3 Evaluation protocols

For each (task, dataset) pair we run TWO protocols:

| Protocol | What | Why |
|---|---|---|
| **Linear Probe** (LP) | Freeze encoder, train linear classifier on train set | Cleanest signal of encoder representation quality |
| **Full Finetune** (FF) | Train encoder + classifier end-to-end | Realistic deployment |

**5 seeds × mean ± std** for both protocols.

### 2.4 Hyperparameters (downstream)

| Param | Value |
|---|---|
| Classifier | 1-layer MLP (linear) |
| Optimizer | AdamW (lr=1e-3 for node classif, 1e-2 for link pred, matching IHP) |
| Epochs | 200 (node), 30 (link pred) |
| Early stop | patience=50 on val metric |

---

## 3. Baselines

### 3.1 Same-backbone / same-data ablation (proves the pretraining value)

| Method | Backbone | Pretrain | Eval |
|---|---|---|---|
| Scratch | UnifiedHypergraphEncoder | none | LP, FF |
| Single-domain pretrain | UnifiedHypergraphEncoder | pretrain on 1 domain only | LP, FF |
| Multi-domain pretrain (Ours) | UnifiedHypergraphEncoder | full 8-task, 4-domain | LP, FF |

### 3.2 Cross-backbone baselines

| Method | Where | What |
|---|---|---|
| HGNN | DHG | Standard hypergraph conv |
| HGNN+ | DHG | Residual version |
| HNHN | DHG | Hypergraph network with node-edge separation |
| UniGCN | DHG | Unified GCN for hypergraphs |
| HyperGCL | DHG | Self-supervised baseline |

### 3.3 Pretrain-method comparison (to validate our pretraining design)

| Method | Source | Note |
|---|---|---|
| IP+Finetune (single-domain pretrain) | our impl | Each dataset pretrained alone, then fine-tuned |
| Hyper-FM-style | our impl of Hyper-FM simplification (HGNN backbone + HyperGCL + bond vertices) | Demonstrates neg_sam_v2 > Hyper-FM-style at same setting |
| GFSE-style | our impl of GFSE on hypergraphs (RWPE + 4 self-supervised tasks) | If we have time |

### 3.4 (Optional) Cross-method comparison with published numbers

Cite Hyper-FM's C-way-1-shot numbers and IHP's link-prediction numbers with explicit protocol disclosure.

---

## 4. Tables and Figures Plan

### Main paper

| ID | Content | Purpose |
|---|---|---|
| **Table 1** | Pretraining datasets, domains, statistics | Show coverage |
| **Table 2** | Pretraining tasks (8) with formula, weight, what-it-teaches | Justify our design |
| **Table 3** | In-domain node classification: scratch vs pretrain (LP + FF) | Show pretrain helps on familiar domains |
| **Table 4** | OOD node classification: scratch vs pretrain | Show pretrain generalizes to unseen domains |
| **Table 5** | Hyperedge prediction (in-domain + OOD): scratch vs pretrain | Show pretrain helps on hypergraph-specific task |
| **Table 6** | Link prediction (rec datasets): scratch vs pretrain vs IHP cited | Show link-pred gain |
| **Table 7** | Ablation: removing each pretrain task | Justify each of the 8 tasks |
| **Table 8** | Ablation: number of pretrain domains (1/2/3/4) | Support GFSE-like scaling claim |
| **Table 9** | Comparison vs HGNN / HNHN / UniGCN / HyperGCL | Standard baselines |
| **Figure 1** | Loss trajectories of 8 tasks over epochs | Show pretrain converges |
| **Figure 2** | Hyperedge prediction: precision-recall curves | Visualize gain |

### Appendix (optional)

- Detailed per-dataset breakdown
- Full hyperparameter sweep
- Per-seed variance tables
- Negative-sampling mode ablation (replace / overlap / random)

---

## 5. Expected Outcomes & Risk Analysis

### What success looks like

| Eval | Pass criterion |
|---|---|
| Linear probe, in-domain node classif | Pretrained > scratch by ≥ +2% (Acc) |
| Linear probe, OOD node classif | Pretrained > scratch by ≥ +1% |
| Full finetune, in-domain | Pretrained ≥ scratch (don't hurt) |
| Hyperedge prediction | Pretrained > scratch by ≥ +3% (Recall@10) |
| Link prediction | Pretrained > scratch by ≥ +2% (Recall@10) |
| Ablation | Removing any of the 8 tasks drops Acc by ≥ +0.5% |

### Risks

| Risk | Mitigation |
|---|---|
| Pretrain loss doesn't drop | Check bf16 autocast, gradient clip, lr |
| In-domain gains masked by feature noise | Use LP (linear probe) as primary signal |
| OOD gains tiny | Consider longer pretrain (300 → 500 epochs) or richer augmentation |
| Hyperedge prediction unstable | Use the same embedding but report recall@K with K ∈ {5, 10, 20} |
| Datasets missing from cache | Document required downloads (`dhg` auto-caches on first use) |

---

## 6. Files and Scripts to Create (after approval)

If approved, I will create the following files **without overwriting existing ones**:

```
configs/
├── pretrain_neg_sam_v2_final.yaml        (8 datasets / 4 domains, with all AMP + grad-clip)
├── eval_in_domain_node.yaml              (Table 3 config)
├── eval_ood_node.yaml                    (Table 4 config)
├── eval_hyperedge.yaml                   (Table 5 config)
├── eval_link_pred.yaml                   (Table 6 config)
├── ablation_remove_<task>.yaml           (Table 7: 8 ablation configs)
└── ablation_num_domains.yaml             (Table 8: 1/2/3/4 domain configs)

scripts/
├── eval_all_v2.sh                        (orchestrator; non-destructive — leaves scripts/eval_all.sh alone)
├── run_eval_in_domain_node.py
├── run_eval_ood_node.py
├── run_eval_hyperedge.py
├── run_eval_link_pred.py
├── run_ablation_tasks.py
└── run_ablation_domains.py

trainers/
└── pretrain_trainer_neg_sam_v2_final.py  (only if needed; otherwise reuse existing)

docs/
└── EXPERIMENTAL_DESIGN.md                (this file)
```

All new files use names with `_v2_final` or `eval_*` prefix to avoid collision with existing ones. The old `scripts/eval_all.sh` and the original `configs/pretrain_neg_sam_v2.yaml` stay untouched.

---

## 7. Summary one-pager

| Stage | What |
|---|---|
| Pretrain | 8 datasets × 4 domains, 8 self-supervised tasks (MNP, HR, MC, CL, DA, Orth, PDP, MGA), quality-aware sampling |
| Downstream | 3 task types: node classification, hyperedge prediction, link prediction |
| Eval datasets | in-domain (cora/citeseer/pubmed/coauthorship_cora/cooking_200) + OOD (house_committees/walmart_trips/movielens_1m) |
| Splits | 60/20/20 random (classification); 80/10/10 (link); per-task-tuned |
| Protocols | Linear probe (clean) + Full finetune (realistic) |
| Seeds | 5 × mean ± std |
| Baselines | scratch, IP-Finetune, HGNN/HGNN+/HNHN/UniGCN, HyperGCL, reimpl Hyper-FM |
| Success | Pretrain > scratch on both LP and FF, on both in-domain and OOD, on both classification and hyperedge prediction |
