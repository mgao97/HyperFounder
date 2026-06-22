# DHG-Bench Migration + Experiment Protocol Alignment (Design)

## Goal

Migrate the current HyperFounder training/evaluation pipeline from `Python-EasyGraph` datasets to **DHG / DHG-Bench** datasets and make the experiments strictly follow the protocol in:

- [实验参数配置.md](file:///Users/santa/Desktop/Hypergraph%20Structural%20Learning/%E5%AE%9E%E9%AA%8C%E5%8F%82%E6%95%B0%E9%85%8D%E7%BD%AE.md)

The target is a reproducible benchmark-style workflow covering:

1. Node-level semi-supervised classification
2. Edge-level / recommendation evaluation
3. Graph-level hypergraph classification
4. A multi-domain pretraining corpus that mixes datasets from multiple domains

The project should **switch to DHG only** (no EasyGraph fallback).

## Non-Goals

- Replacing the HyperGT-style encoder or multitask pretraining objective.
- Implementing distributed training.
- Implementing sparse attention kernels inside the Transformer.

## Current State (Constraints)

- The repo currently loads datasets via `easygraph.datasets.hypergraph.*` and converts them into `SimpleHypergraph`.
- Downstream evaluation is currently only node classification.
- Some YAML keys are present but unused (e.g., `warmup_epochs`, `prototype_refresh_interval`).
- The environment currently does not have OpenAI configured; this is unrelated to DHG-Bench migration.

## Design Overview

We keep the existing model/trainer structure and replace the data + protocol layer:

- Keep:
  - `SimpleHypergraph` transport object.
  - HyperGT-style encoder + pretext tasks.
  - Subhypergraph minibatch sampling for large graphs.
- Replace / Add:
  - Replace dataset registry + loader with **DHG dataset registry + adapter**.
  - Add missing downstream tracks (recommendation + graph classification).
  - Add early stopping, multi-seed reporting, and protocol-required metric logging.
  - Update configs to match protocol-default hyperparameters.

## Dataset Layer

### Dataset Registry

Add a DHG-backed registry that stores:

- `name`: stable dataset name used in configs (e.g., `cora`, `citeseer`, `pubmed`, `dblp`, `acm`, `imdb`, `yelp`, `gowalla`, `tmall`, …)
- `task_type`: one of `{node_cls, rec, graph_cls}`
- `domain`: one of `{citation, academic, recommendation, document, biological, brain, ...}` used for multi-domain pretraining sampling
- `loader`: a callable that instantiates the dataset from `dhg.data` (or `dhg_bench` if required by the package)

### Adapter Contract (DHG → SimpleHypergraph)

Implement a single conversion function:

- Input: a DHG dataset object (supports `d["edge_list"]`, `d["labels"]`, `d["features"]` when available, plus official masks/splits when provided)
- Output: `SimpleHypergraph` with:
  - `hyperedges`: DHG `edge_list` normalized to `List[List[int]]`
  - `x`: node feature matrix
  - `node_labels`: node labels
  - `node_train_mask/node_val_mask/node_test_mask`: **official masks only**
  - `metadata`: protocol-required stats (see below)

### Split Policy (Strict)

To align with the protocol:

- If official splits are available in the DHG dataset object, use them.
- Do **not** create random splits as a fallback.
- If a dataset does not provide required official split artifacts for a given task, the loader must error with a clear message.

### Dataset Statistics Logging (Required)

For each dataset instance loaded, compute and store:

- `num_nodes`, `num_hyperedges`
- `avg_hyperedge_size`, `max_hyperedge_size`
- `feature_dim`
- `num_classes` (node or graph classes depending on task)
- `train/val/test` sizes (mask counts)

These stats must be persisted into:

- pretraining summary JSON
- downstream evaluation summary JSON

## Config Schema Changes

### Protocol-default Hyperparameters

Update `configs/pretrain.yaml` (and any downstream configs) to protocol defaults:

- Optimizer: AdamW
- Learning rate: 1e-3
- Weight decay: 1e-4
- Hidden dim: 128
- Layers: 4
- Heads: 8
- Dropout: 0.3
- Pretrain max epochs: 300
- Early stopping patience: 50
- Multi-seed: 3–5 seeds (default 3 to match current code, configurable)

### Data Section

Replace EasyGraph dataset names with DHG registry names.

Add fields:

- `data.backend: dhg`
- `data.datasets: [ ... ]` (list of DHG dataset names)
- `data.domain_map: {dataset_name: domain}` (explicit mapping to keep multi-domain corpus stable)
- `data.cache_dir`: forwarded to DHG `data_root` if supported

For “full protocol”, `data.datasets` should include at least:

- Node-level: `cora`, `citeseer`, `dblp`, `acm`, `imdb` (exact DHG names must match the installed package API)
- Recommendation: `yelp`, `gowalla`, `tmall`
- Graph-level: document/biological/brain datasets when available in DHG-Bench

The exact dataset identifiers must be implemented based on the installed DHG/DHG-Bench API (verified by a small “list datasets” probe script during implementation).

### Training Section

Add early stopping fields:

- `training.max_epochs` (or reuse `epochs` as max epochs)
- `training.early_stopping.patience`
- `training.early_stopping.metric` (task dependent)
- `training.num_seeds`

Note: `prototype_refresh_interval` and `warmup_epochs` should either be implemented or removed from configs for strictness.

## Training / Evaluation Protocol Alignment

### Multi-Seed Runs (Required)

All reported results must run with `training.num_seeds` (default 3, allow 5):

- Record mean ± std for the main metrics.
- Persist per-seed metrics for auditability.

### Early Stopping (Required)

Add early stopping to all downstream tasks:

- Node: validation accuracy (and compute macro-F1 for reporting)
- Recommendation: validation HR@10 / NDCG@10 (follow the dataset’s official split protocol)
- Graph-level: validation accuracy or AUC depending on dataset

For pretraining, use:

- primary: smoothed pretext loss (as a proxy) OR a small held-out downstream proxy if explicitly requested

### Metrics (Required)

Implement and report:

- Node: Accuracy, Macro-F1
- Recommendation: HR@5/10, NDCG@5/10
- Graph: Accuracy, and AUC when appropriate

### Fairness Constraint (Protocol)

When comparing pretrained vs scratch:

- keep all downstream hyperparameters identical
- only change encoder initialization

## Code Changes (Planned)

### New / Replaced Modules

- Replace `utils/dataset_registry.py` with a DHG-backed registry.
- Replace `utils/easyhypergraph_datasets.py` with `utils/dhg_datasets.py`:
  - load dataset objects from DHG
  - enforce official splits only
  - convert to `SimpleHypergraph`
  - compute dataset statistics

### Trainers

- Extend downstream trainer to support:
  - node classification (existing)
  - recommendation evaluation (new)
  - graph classification (new)
- Add:
  - early stopping
  - multi-seed aggregation to match protocol

### Dependencies

- Remove `Python-EasyGraph` from requirements.
- Add `dhg-bench` (and any required underlying `dhg` package if it is not a transitive dependency).

## Verification Plan

The migration is considered correct only if:

1. `configs/pretrain.yaml` uses DHG/DHG-Bench datasets and protocol-default hyperparameters.
2. Loader refuses to run if a dataset lacks official split artifacts needed for the specified task.
3. Node-level transfer runs and reports accuracy + macro-F1 with mean ± std over multi-seeds.
4. Recommendation evaluation runs and reports HR@K / NDCG@K.
5. Graph-level evaluation runs and reports accuracy/AUC.
6. Summaries contain required dataset stats and training hyperparameters for audit.

