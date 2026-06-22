# HyperGT Subhypergraph Minibatch Design

## Objective

Enable the current HyperGT pretraining pipeline to scale to large hypergraphs such as:

- `walmart_trips`
- `yelp_restaurant`

by replacing the current whole-graph pretraining path with a subhypergraph/minibatch sampling strategy.

The selected approach is:

- `Approach C`
  - online hyperedge-centered seeded subhypergraph minibatch as the default path
  - optional offline subhypergraph pool for very large hypergraphs
  - optional extra token capping before the encoder for extreme cases

This design keeps the current HyperGT backbone and multi-task pretraining objective, but changes the training unit from full graphs to sampled local subhypergraphs.

## Scope

### In Scope

- add hyperedge-centered seeded subhypergraph minibatch sampling
- change pretraining from whole-graph iteration to sampled subhypergraph steps
- support optional offline pre-sampling of subhypergraphs for very large datasets
- keep current pretraining tasks and adapt them to sampled subhypergraphs
- add config controls for sampling budgets
- keep downstream transfer code unchanged unless required by new checkpoint metadata

### Out of Scope

- rewriting the backbone to sparse attention
- adding true layerwise sparse attention kernels
- changing the default downstream evaluation task from node classification
- implementing distributed training in this iteration

## Why Hyperedge-Centered Sampling

Hypergraphs are built around high-order incidence structure, so the most natural local unit is a hyperedge and its overlapping neighborhood.

This approach preserves:

- incident node-hyperedge pairs for `L_struct`
- overlapping hyperedges for local structure
- motif patterns built from intersecting hyperedges
- community-like local overlap regions

Compared with edge-chunk batching, hyperedge-centered expansion keeps local structure more coherent. Compared with a full hybrid sampler, it is simpler and safer as a first scalable implementation.

## Training Unit

### Previous Unit

The previous pretraining loop used:

- one full hypergraph per training step

This fails on large datasets because the current backbone computes dense:

- node-node structural overlap
- edge-edge structural overlap
- node attention bias
- edge attention bias

### New Unit

The new pretraining loop uses:

- one minibatch of sampled subhypergraphs per training step

Each sampled subhypergraph becomes a normal HyperGT input instance with:

- local node features
- local hyperedge list
- local incidence matrix
- mappings back to the original large graph

## Sampling Strategy

### Default Online Sampling

For each training step:

1. sample graphs from available domains
2. sample one or more seed hyperedges from each chosen graph
3. expand each seed into a local subhypergraph
4. cap by `max_nodes` and `max_edges`
5. pack the resulting sampled subhypergraphs as the batch for that step

### Hyperedge-Centered Expansion

Given a seed hyperedge:

1. include the seed hyperedge
2. include all incident nodes
3. find neighboring hyperedges that overlap with current nodes
4. add neighbors until the hop budget or size budget is reached
5. induce the local node and hyperedge subset

Recommended first version:

- expansion basis: overlapping hyperedges through shared nodes
- depth: `1` or `2` hops
- hard caps: `max_nodes`, `max_edges`

### Budget Control

Sampling must stop when any of the following is reached:

- `max_nodes`
- `max_edges`
- `expansion_hops`

If the frontier is larger than the remaining budget:

- prefer hyperedges with larger overlap to the current sampled node set
- break ties randomly using the configured seed

## Offline Subhypergraph Pool

### Purpose

For extremely large graphs, online expansion can still be expensive. The project should support an optional subhypergraph pool.

### Behavior

At dataset preparation time:

1. sample many seed hyperedges
2. expand them into local subhypergraphs
3. store the sampled subhypergraph metadata in memory or cache

During training:

- sample from the subhypergraph pool instead of expanding from scratch every step

### Pool Policy

The first version should support:

- per-graph pool size
- refresh on startup only
- optional periodic rebuild later, but not required in this version

## Internal Representation

### Sampled Subhypergraph Object

Reuse `SimpleHypergraph` as the main transport object, but extend its metadata to include:

- `parent_graph_name`
- `parent_dataset_name`
- `global_node_ids`
- `global_edge_ids`
- `seed_edge_ids`
- `sampling_depth`
- `sampling_strategy`

This keeps compatibility with the current encoder interface while preserving provenance back to the original graph.

## New Modules

### `utils/minibatch_sampling.py`

This new file should contain focused sampling utilities:

- `sample_seed_hyperedges()`
- `expand_hyperedge_centered_subhypergraph()`
- `induce_sampled_subhypergraph()`
- `build_subhypergraph_pool()`
- `sample_subhypergraph_batch()`

### Responsibilities

- sampling policy lives here
- graph-to-subhypergraph induction lives here
- pool creation and sampling live here
- trainer should call this module instead of implementing graph expansion inline

## Trainer Changes

### `pretrain_trainer.py`

The trainer should move from graph iteration to step-based minibatch sampling.

New flow:

1. load full dataset graphs
2. create sampler state for each graph
3. optionally build subhypergraph pools for large graphs
4. for each epoch and step:
   - sample a minibatch of subhypergraphs
   - run encoder and heads on each sampled subhypergraph
   - aggregate losses across the minibatch
   - update the optimizer

### Domain Balance

Each step should sample subhypergraphs across domains rather than repeatedly favoring one large graph.

The first version should support:

- configurable `domains_per_step`
- configurable `subhypergraphs_per_domain`

This keeps the cross-domain pretraining objective meaningful.

## Task Adaptation

All pretraining tasks remain, but their working scope becomes the sampled subhypergraph.

### `struct`

- unchanged in form
- computed only on incident pairs inside the sampled subhypergraph

### `node`

- predict node pseudo-labels only for sampled nodes
- labels should be inherited or recomputed from the sampled view

### `edge`

- positive edges are sampled subhypergraph hyperedges
- negative edges are generated within the sampled node universe

### `motif`

- motifs are sampled only inside the local subhypergraph
- this becomes cheaper and still structurally meaningful

### `community`

- communities are detected only in the sampled local overlap structure

### `global`

- first version becomes subhypergraph-level contrastive learning
- two augmentations are generated from the same sampled subhypergraph

This keeps the current loss name for compatibility, but the semantics in documentation should explicitly state that it is local-subhypergraph contrastive pretraining rather than whole-graph contrastive pretraining.

### `cross`

- prototypes are built from sampled motif, community, and subhypergraph readout embeddings
- prototype refresh should operate on sampled embeddings, not only full-graph embeddings

## Pseudo-Label Policy

The first scalable version should avoid expensive full-graph recomputation during every step.

Recommended policy:

- compute or refresh pseudo-labels on sampled subhypergraphs
- keep per-subhypergraph temporary caches only for the current step
- do not require a full-graph cache rebuild every minibatch

For offline pool mode:

- motif/community caches may be stored per pooled subhypergraph

## Config Changes

### New Training Fields

Add sampling controls under `training` or `data`.

Recommended fields:

```yaml
training:
  steps_per_epoch: 32
  minibatch:
    domains_per_step: 2
    subhypergraphs_per_domain: 2
    max_nodes: 256
    max_edges: 128
    expansion_hops: 2
    seed_edges_per_subhypergraph: 1
    use_subhypergraph_pool: true
    subhypergraph_pool_size: 512
    large_graph_node_threshold: 5000
```

### Behavior

- small and medium graphs may still use online sampling only
- graphs above `large_graph_node_threshold` should default to pool mode when enabled

## Encoder and Backbone Policy

The encoder interface should remain as stable as possible:

- sampled subhypergraphs are still passed as normal `SimpleHypergraph` instances

The backbone should not require architectural changes in the first version.

Optional safeguard:

- if a sampled subhypergraph still exceeds a safe token threshold, apply an additional pre-encoder token cap through sampling-time truncation rather than modifying Transformer layers

## Verification Plan

The work is complete only when all of the following pass:

- `walmart_trips` can start pretraining without attempting whole-graph dense attention
- `yelp_restaurant` can start pretraining without attempting whole-graph dense attention
- `run_pretrain.py` works with online hyperedge-centered minibatch sampling
- pooled mode works on at least one large dataset
- loss history is written as before
- checkpoint saving still works
- diagnostics show no new Python errors

## Files To Modify

- `trainers/pretrain_trainer.py`
- `models/pretext_tasks.py`
- `utils/sampling.py`
- `configs/pretrain.yaml`
- `README.md`

## Files To Add

- `utils/minibatch_sampling.py`

## Constraints

- keep comments concise and in English
- avoid deep nesting in the sampler implementation
- prefer explicit metadata over hidden coupling
- keep old small-graph workflows working through the same trainer where practical
- do not silently fall back to whole-graph mode for very large graphs
