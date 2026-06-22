# Downstream Protocol Alignment + Timing Statistics (Design)

## Goal

Extend the current DHG-based evaluation pipeline so that it aligns with the full protocol in three areas:

1. Held-out domain reporting for node-level transfer
2. Recommendation evaluation with HR@K / NDCG@K
3. Graph-level evaluation support
4. Train/eval timing statistics saved into result files

This design builds on the existing DHG migration and does not change the HyperGT encoder or the pretraining objective.

## Scope

### In Scope

- Align `configs/finetune_node.yaml` with the supported held-out domains and datasets.
- Add a recommendation evaluation track using DHG recommendation datasets already available in the loader.
- Add a graph-level evaluation track interface and config path.
- Record timing statistics in saved JSON/CSV results.

### Out of Scope

- Changing the sampling logic of pretraining.
- Implementing external benchmark runners.
- Replacing the current node fine-tuning strategy.

## Current Constraints

- The current `FinetuneTrainer` only supports node classification.
- Recommendation datasets are already mapped into `SimpleHypergraph`, but only as data transport objects.
- Graph-level datasets may not be exposed by the current DHG install; the code must therefore support them cleanly and fail clearly when no compatible dataset is configured.
- Timing should follow the user decision: **record train + eval only**, excluding dataset download/load time.

## Design

## 1. Node-Level Held-Out Alignment

The node transfer config should represent only datasets that:

- are exposed by the DHG registry,
- are intended for node-level evaluation,
- provide official `train_mask`, `val_mask`, and `test_mask`.

Supported held-out domains should be expressed through `domain_map` and aliases:

- `citation`
- `academic`
- `document`
- `recommendation`

The node transfer result JSON should include:

- `heldout_domain`
- `evaluated_datasets`
- `node_accuracy`
- `node_accuracy_std`
- `node_macro_f1`
- `node_macro_f1_std`
- `finetune_train_time_sec`
- `finetune_eval_time_sec`

## 2. Recommendation Track

### Data Interpretation

For recommendation datasets:

- items are represented as nodes,
- users are represented as hyperedges,
- each user hyperedge contains the interacted training items.

The adapter already stores:

- `train_adj_list`
- `test_adj_list`
- `num_users`
- `num_items`

### Fine-Tuning Objective

Use a pairwise ranking objective on train interactions:

- user embedding: pooled or token-aligned hyperedge embedding
- item embedding: node embedding
- score: dot product between user and item embedding
- loss: BPR-style pairwise ranking with negative sampling

### Validation / Early Stopping

Because the current DHG recommendation objects may only provide train/test lists:

- keep the official test list untouched,
- derive a deterministic validation target from training interactions when needed for early stopping,
- use the remaining training interactions for optimization.

This preserves a stable test protocol while still allowing early stopping.

### Metrics

For each dataset and aggregated summary:

- `hr@5`
- `hr@10`
- `ndcg@5`
- `ndcg@10`
- `finetune_train_time_sec`
- `finetune_eval_time_sec`

## 3. Graph-Level Evaluation Track

### Dataset Contract

Graph-level evaluation requires a dataset that yields:

- multiple hypergraph instances,
- per-graph labels,
- train/val/test graph splits or equivalent split artifacts.

### Trainer Behavior

- If configured datasets satisfy the graph-level contract, run graph classification using graph embeddings from the encoder.
- If no configured dataset satisfies the contract, raise a clear error explaining that the current DHG environment does not expose compatible graph-level datasets.

### Metrics

When supported by the dataset:

- `graph_accuracy`
- `graph_accuracy_std`
- `graph_auc`
- `graph_auc_std`
- `finetune_train_time_sec`
- `finetune_eval_time_sec`

If AUC is not applicable, store `null` or omit it consistently.

## 4. Timing Statistics

The user-selected rule is:

- record **train + eval only**
- exclude dataset download/load time

### Pretraining

Persist into pretraining summary:

- `pretrain_train_time_sec`

### Downstream

Persist into downstream summaries:

- `finetune_train_time_sec`
- `finetune_eval_time_sec`

Timing should be measured with `time.perf_counter()` around:

- the optimization loop,
- the evaluation loop.

Data loading and adapter conversion must happen before timing starts.

## 5. CLI / Config Shape

Keep the existing transfer entry style:

- `scripts/run_transfer.py` for node and graph tasks

Add a recommendation entry if separation keeps the code cleaner:

- `scripts/run_recommendation.py`

Recommended configs:

- `configs/finetune_node.yaml`
- `configs/finetune_rec.yaml`
- `configs/finetune_graph.yaml`

## 6. Verification Plan

The implementation is considered complete only if:

1. Node transfer runs for the supported held-out domains and saves timing statistics.
2. Recommendation evaluation runs on at least one DHG recommendation dataset and saves HR/NDCG plus timing statistics.
3. Graph evaluation either runs on a compatible DHG graph dataset or raises a clear contract error.
4. Result files contain the requested timing fields.
5. The code keeps training behavior unchanged outside the new downstream additions.

