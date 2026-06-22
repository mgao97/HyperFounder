# Training Parameters Audit

This document lists the parameters that control training, sampling, and evaluation, together with their code-level consumption points.

## Pretraining

### Config Files

- [pretrain.yaml](file:///Users/santa/Desktop/HyperFounder/configs/pretrain.yaml): cross-domain pretraining over multiple EasyHypergraph datasets.
- [pretrain_large_graph.yaml](file:///Users/santa/Desktop/HyperFounder/configs/pretrain_large_graph.yaml): large-graph stress path (single dataset, minibatch sampling + pool).

### Model Parameters (`model.*`)

- `model.input_dim`
  - Used by [load_domain_graphs](file:///Users/santa/Desktop/HyperFounder/utils/easyhypergraph_datasets.py#L108-L125) → `target_dim` → [load_easyhypergraph_sample](file:///Users/santa/Desktop/HyperFounder/utils/easyhypergraph_datasets.py#L64-L105) to resize/fallback node features.
  - Used by [PretrainTrainer.__init__](file:///Users/santa/Desktop/HyperFounder/trainers/pretrain_trainer.py#L24-L52) to build [UnifiedHypergraphEncoder](file:///Users/santa/Desktop/HyperFounder/models/encoder.py#L13-L34).
- `model.hidden_dim`, `model.dropout`, `model.num_layers`, `model.num_heads`, `model.spectral_dim`
  - Used by [PretrainTrainer.__init__](file:///Users/santa/Desktop/HyperFounder/trainers/pretrain_trainer.py#L24-L52) to construct the encoder and task heads.

### Task Head Parameters (`tasks.*`)

- `tasks.node_clusters`, `tasks.motif_clusters`, `tasks.community_clusters`, `tasks.prototype_clusters`
  - Used by [PretrainTrainer.__init__](file:///Users/santa/Desktop/HyperFounder/trainers/pretrain_trainer.py#L24-L52) to size heads and pseudo-label cluster counts.
  - Used by [PretrainTrainer._build_subhypergraph_task_cache](file:///Users/santa/Desktop/HyperFounder/trainers/pretrain_trainer.py#L94-L122) and [refresh_cross_domain_prototypes](file:///Users/santa/Desktop/HyperFounder/trainers/pretrain_trainer.py#L123-L153).

### Core Training Parameters (`training.*`)

- `training.seed`
  - Used by [run_pretrain.py](file:///Users/santa/Desktop/HyperFounder/scripts/run_pretrain.py#L21-L29) to set global RNG seeds.
  - Used by [PretrainTrainer](file:///Users/santa/Desktop/HyperFounder/trainers/pretrain_trainer.py#L24-L75) for pool building, domain schedule, and per-step sampling seeds.
- `training.device`
  - Used by [PretrainTrainer.__init__](file:///Users/santa/Desktop/HyperFounder/trainers/pretrain_trainer.py#L24-L31).
- `training.epochs`, `training.steps_per_epoch`
  - Used by [PretrainTrainer.train](file:///Users/santa/Desktop/HyperFounder/trainers/pretrain_trainer.py#L166-L238).
  - Effective number of optimization steps per run is `epochs * steps_per_epoch` (each step processes a minibatch of sampled subhypergraphs).
- `training.lr`, `training.weight_decay`
  - Used by [PretrainTrainer.__init__](file:///Users/santa/Desktop/HyperFounder/trainers/pretrain_trainer.py#L47-L52) to build AdamW.
- `training.motif_budget`
  - Used by [PretrainTrainer._build_subhypergraph_task_cache](file:///Users/santa/Desktop/HyperFounder/trainers/pretrain_trainer.py#L94-L122) to bound motif pseudo-label sampling.
  - Used by [compute_pretraining_losses](file:///Users/santa/Desktop/HyperFounder/models/pretext_tasks.py#L38-L57) as encoder `motif_budget`.
- `training.feature_mask_rate`, `training.edge_dropout_rate`
  - Used only by `global` task augmentation in [compute_pretraining_losses](file:///Users/santa/Desktop/HyperFounder/models/pretext_tasks.py#L102-L121) → [augment_hypergraph](file:///Users/santa/Desktop/HyperFounder/utils/sampling.py#L167-L176).
- `training.output_dir`
  - Used by [PretrainTrainer.__init__](file:///Users/santa/Desktop/HyperFounder/trainers/pretrain_trainer.py#L53-L58) to place checkpoints/logs/results under `outputs/`.

### Sampling / Minibatch Parameters (`training.minibatch.*`)

These parameters control what a single training step consumes (a minibatch of sampled subhypergraphs).

- `training.minibatch.domains_per_step`
  - Used by [PretrainTrainer._build_domain_schedule](file:///Users/santa/Desktop/HyperFounder/trainers/pretrain_trainer.py#L76-L92) to build a per-epoch list of “preferred domains” per step.
- `training.minibatch.subhypergraphs_per_domain`
  - Used by [sample_subhypergraph_batch](file:///Users/santa/Desktop/HyperFounder/utils/minibatch_sampling.py#L171-L204) to pick how many subhypergraphs to sample for each chosen domain.
- `training.minibatch.max_nodes`, `training.minibatch.max_edges`, `training.minibatch.expansion_hops`
  - Used by [sample_online_subhypergraph](file:///Users/santa/Desktop/HyperFounder/utils/minibatch_sampling.py#L137-L150) / [expand_hyperedge_centered_subhypergraph](file:///Users/santa/Desktop/HyperFounder/utils/minibatch_sampling.py#L79-L134).
- `training.minibatch.seed_edges_per_subhypergraph`
  - Used by [sample_online_subhypergraph](file:///Users/santa/Desktop/HyperFounder/utils/minibatch_sampling.py#L137-L150).
- `training.minibatch.use_subhypergraph_pool`, `training.minibatch.subhypergraph_pool_size`, `training.minibatch.large_graph_node_threshold`
  - Used by [should_use_subhypergraph_pool](file:///Users/santa/Desktop/HyperFounder/utils/minibatch_sampling.py#L153-L156) and [build_subhypergraph_pool](file:///Users/santa/Desktop/HyperFounder/utils/minibatch_sampling.py#L159-L168).
  - Pool is created once at trainer init via [PretrainTrainer._build_pool_cache](file:///Users/santa/Desktop/HyperFounder/trainers/pretrain_trainer.py#L63-L74).

### Loss Weights (`training.loss_weights.*`)

- `training.loss_weights.{struct,node,edge,motif,community,global,cross}`
  - Used by [compute_pretraining_losses](file:///Users/santa/Desktop/HyperFounder/models/pretext_tasks.py#L58-L134).

### Present but Unused (Audit Flags)

- `training.prototype_refresh_interval`
- `training.warmup_epochs`

These keys exist in the YAML configs but are currently not referenced anywhere in the Python code (repo-wide search shows no usage).

## Downstream Transfer (Node Classification)

### Config File

- [finetune_node.yaml](file:///Users/santa/Desktop/HyperFounder/configs/finetune_node.yaml)

### Key Parameters

- `training.pretrained_checkpoint`
  - Loaded by [FinetuneTrainer._build_encoder](file:///Users/santa/Desktop/HyperFounder/trainers/finetune_trainer.py#L26-L46) (shape-compatible partial load).
- `training.finetune_epochs`, `training.lr`
  - Used by [FinetuneTrainer._run_node_task](file:///Users/santa/Desktop/HyperFounder/trainers/finetune_trainer.py#L77-L103).
- `data.datasets`, `data.domain_map`
  - Used by [FinetuneTrainer._select_dataset_names](file:///Users/santa/Desktop/HyperFounder/trainers/finetune_trainer.py#L52-L65) and [load_domain_graphs](file:///Users/santa/Desktop/HyperFounder/utils/easyhypergraph_datasets.py#L108-L125).

