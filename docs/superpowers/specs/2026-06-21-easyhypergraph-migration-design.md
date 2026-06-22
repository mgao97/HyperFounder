# EasyHypergraph Migration Design

## Objective

Migrate the current project from a synthetic hypergraph data path to an EasyHypergraph-native dataset stack while preserving the project's main research direction:

- keep the current HyperGT-style encoder as the primary model
- keep the multi-task pretraining objective
- replace the synthetic data generator and DHG-compatible assumptions
- load real datasets supported by EasyHypergraph-native workflows
- remove obsolete synthetic-only logic and stale dataset assumptions

This migration is a full data and training-stack migration, not a small adapter patch.

## Scope

### In Scope

- replace synthetic dataset generation with real dataset loading
- introduce an EasyHypergraph-native dataset registry
- reorganize pretraining and transfer around real datasets and real domains
- update configs to use dataset names instead of synthetic graph size parameters
- remove synthetic-only code paths that are no longer needed
- keep the HyperGT encoder as the main model implementation

### Out of Scope

- porting the whole project into a clone of the benchmark repository
- replacing HyperGT with EasyHypergraph benchmark baselines as the default model
- implementing datasets that require DHG-only loaders
- adding explicit sparse attention kernels in this migration

## Migration Strategy

### Recommended Strategy

Use EasyHypergraph-native datasets as the only supported source of hypergraph data in this project and preserve the current project-level interfaces:

- `run_pretrain.py`
- `run_transfer.py`
- `run_ablation.py`

The project keeps its current purpose as a HyperGT pretraining scaffold, but its data source becomes real datasets instead of synthetic graphs.

### Why This Strategy

- it satisfies the requirement to move away from synthetic and DHG-style assumptions
- it avoids rewriting the project into a benchmark-only runner
- it preserves the current encoder, loss design, and experiment entry points
- it allows later addition of EasyHypergraph baseline models without forcing them now

## Dataset Policy

### Supported Dataset Scope

This migration only targets EasyHypergraph-native datasets and intentionally excludes DHG-only dataset dependencies.

The first implementation batch should support the datasets that are most aligned with EasyHypergraph-native usage and practical coverage:

- `walmart_trips`
- `trivago_clicks`
- `senate_committees`
- `house_committees`
- `cocitation_cora`
- `cocitation_citeseer`
- `cocitation_pubmed`

### Domain Mapping

Datasets are grouped into broad domains for pretraining and held-out transfer:

- `citation`
  - `cocitation_cora`
  - `cocitation_citeseer`
  - `cocitation_pubmed`
- `commerce`
  - `walmart_trips`
  - `trivago_clicks`
- `political`
  - `senate_committees`
  - `house_committees`

The project should use domain names derived from dataset membership instead of synthetic domain buckets.

## Data Representation

### Unified Internal Sample

The project should preserve a single internal hypergraph sample contract used by the model and trainers. The current `SimpleHypergraph` structure can be kept if it remains useful, but it must stop implying synthetic generation.

Each sample must expose:

- node feature matrix
- hyperedge list or incidence information
- dataset name
- domain name
- available labels for node, edge, or graph tasks
- metadata needed for split generation and caching

If a real dataset does not provide some labels:

- node labels should be used when available
- graph or edge labels should only be enabled for tasks that the dataset supports
- unsupported downstream tasks should be skipped or rejected explicitly instead of fabricated

## Architecture Changes

### New Data Modules

Add focused data modules instead of keeping dataset logic inside one file:

- `utils/dataset_registry.py`
  - maps dataset names to loader functions and metadata
- `utils/easyhypergraph_datasets.py`
  - loads EasyHypergraph-native datasets and converts them into the internal sample format
- `utils/splits.py`
  - builds train, validation, and test splits for real datasets

### Existing Module Changes

- `utils/hypergraph.py`
  - keep only reusable hypergraph container utilities
  - remove synthetic graph generation and synthetic domain construction
- `trainers/pretrain_trainer.py`
  - load datasets from config
  - build graph collections from real datasets grouped by domain
- `trainers/finetune_trainer.py`
  - use real dataset splits instead of synthetic held-out graph partitions
- `configs/*.yaml`
  - replace synthetic graph generation knobs with dataset selection and split settings

## Dataset Loading Flow

### Pretraining

1. read dataset names from config
2. load each dataset through the registry
3. convert each dataset into one or more internal hypergraph samples
4. assign domain labels through the domain map
5. build the pretraining graph list
6. cache dataset-level structural artifacts when useful

### Transfer Evaluation

1. select a held-out domain or held-out dataset
2. load only datasets belonging to that target scope
3. build downstream task splits from real labels
4. run finetuning with the current HyperGT encoder and task-specific heads

## Model Policy

### Primary Model

Keep the HyperGT-style dual-token encoder as the primary model and the default option.

The project should not switch its default model to:

- HGNN
- HNHN
- UniGCN
- other benchmark baselines

### Optional Future Extension

Later work may add a model registry for EasyHypergraph benchmark baselines, but that is not required in this migration.

## Task Support Rules

### Pretraining Tasks

Keep the current pretraining tasks:

- `struct`
- `node`
- `edge`
- `motif`
- `community`
- `global`
- `cross`

These tasks now operate on real datasets instead of synthetic graphs.

### Downstream Tasks

The project should support downstream tasks only when the loaded dataset provides compatible labels:

- `node` classification when node labels exist
- `edge` classification only when edge labels exist or can be derived from official dataset targets
- `graph` classification only when graph labels exist

If a requested task is unsupported for a dataset, the pipeline should fail clearly with a useful error message.

## Config Changes

### New Pretraining Config Shape

The synthetic domain section should be removed and replaced by dataset-driven configuration.

Example shape:

```yaml
model:
  name: hypergt
  input_dim: 16
  hidden_dim: 64
  dropout: 0.3
  num_layers: 2
  num_heads: 4
  spectral_dim: 4

data:
  datasets:
    - cocitation_cora
    - cocitation_citeseer
    - walmart_trips
  domain_map:
    cocitation_cora: citation
    cocitation_citeseer: citation
    walmart_trips: commerce
  feature_strategy: dataset
  cache_dir: data/cache

training:
  ...
```

### Feature Strategy

Support explicit feature handling policies:

- `dataset`
  - use native dataset features when provided
- `identity`
  - use identity-like features when feasible for small datasets
- `random`
  - use random fallback features only when the dataset lacks features and no better option exists

The default should prefer dataset-native features.

## Cleanup Policy

Remove the following once the migration is complete:

- synthetic graph generation
- synthetic domain generation
- synthetic labels used only for smoke testing
- README language that claims the project is synthetic-only
- config fields for `num_graphs`, `num_nodes`, and `num_edges`

Do not keep dead compatibility layers if they are no longer needed for the new design.

## Error Handling

The new pipeline should fail clearly for:

- unknown dataset names
- unsupported task and dataset combinations
- missing labels required by a downstream task
- dataset download or parsing failures
- feature shape mismatches

Errors should identify:

- dataset name
- task name
- stage where loading or evaluation failed

## Verification Plan

The migration is complete only when the project passes all of the following:

- at least two EasyHypergraph-native datasets load successfully
- pretraining runs for one smoke-test epoch on real datasets
- node transfer runs successfully on one labeled dataset
- graph transfer runs successfully on one graph-labeled dataset if supported by the selected dataset batch
- no synthetic-only dependency remains in the active training path
- no DHG dependency remains in the active data path
- diagnostics show no new Python errors

## Files To Rewrite

- `utils/hypergraph.py`
- `trainers/pretrain_trainer.py`
- `trainers/finetune_trainer.py`
- `configs/pretrain.yaml`
- `configs/finetune_node.yaml`
- `configs/finetune_edge.yaml`
- `configs/finetune_graph.yaml`
- `README.md`
- `requirements.txt`

## Files To Add

- `utils/dataset_registry.py`
- `utils/easyhypergraph_datasets.py`
- `utils/splits.py`

## Constraints

- keep comments concise and in English
- avoid unnecessary compatibility wrappers
- prefer small focused modules over one large mixed data file
- do not fabricate labels for unsupported real datasets
- keep current CLI entry points stable where possible
