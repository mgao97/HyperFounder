# HyperGT Upgrade Design

## Objective

Upgrade the current v1 HGNN-based hypergraph pretraining scaffold into a Hypergraph Transformer / HyperGT-style encoder implementation under `/Users/santa/Desktop/HyperFounder`.

The upgraded version must:

- replace the HGNN backbone with a dual-token Hypergraph Transformer
- treat node tokens and hyperedge tokens as first-class representations
- inject incidence-based structural encoding into token initialization and attention bias
- add a backbone-level hypergraph structure regularization term
- support six pretraining tasks:
  - node
  - hyperedge
  - motif
  - community
  - global
  - cross-hypergraph
- preserve the existing CLI entry points while upgrading internal semantics
- remove obsolete HGNN-specific logic and unused helper behavior after the migration

## Scope

### In Scope

- rewrite `models/backbone.py` as a Hypergraph Transformer backbone
- rewrite `models/encoder.py` as a dual-stream node/edge encoder
- extend pretraining tasks from five tasks to six tasks plus structure regularization
- update training configs and trainers to use AdamW and Transformer-friendly options
- update downstream finetuning to consume upgraded node, edge, and graph embeddings
- clean up legacy HGNN-only code paths that are no longer used

### Out of Scope

- explicit node-hyperedge cross-attention in the first upgraded version
- full sparse attention kernels or custom CUDA optimization
- replacing the synthetic data adapter with production DHG dataset integration

## Architecture

### Core Design

The new encoder is a HyperGT-style backbone with two token types:

- node tokens
- hyperedge tokens

Each encoder layer updates node tokens and edge tokens with separate self-attention blocks. Node-edge coupling is carried by:

- incidence-based initialization
- structural positional encodings
- relative structural attention bias
- structure regularization on incident node-edge pairs

The first upgraded version intentionally skips explicit node-edge cross-attention to keep the implementation bounded and aligned with the approved plan.

### Token Initialization

#### Node Tokens

Initial node tokens are:

```python
node_tokens = node_projection(x) + node_positional_encoding
```

The node positional encoding uses incidence-derived node structural features:

- log degree
- incident hyperedge size mean
- incident hyperedge size max
- node overlap statistics from `H @ H.T`
- lightweight spectral or random-walk style structural channels derived from `H @ H.T`

#### Hyperedge Tokens

Initial edge tokens are:

```python
edge_tokens = edge_input_projection(mean(projected_incident_nodes)) + edge_positional_encoding
```

The edge positional encoding uses hyperedge structural features:

- edge size
- overlap statistics from `H.T @ H`
- lightweight spectral or random-walk style structural channels derived from `H.T @ H`

### Transformer Block

Each layer contains:

- node self-attention with node-node structural bias
- edge self-attention with edge-edge structural bias
- residual connections
- feed-forward layers
- layer normalization

The implementation should keep node and edge channels symmetric where possible to reduce branching and make later cross-attention insertion easier.

### Structural Bias

The backbone computes relative structural bias matrices:

- `B_vv` for node-node attention
- `B_ee` for edge-edge attention

These are built from pairwise structural signals such as:

- co-membership counts
- overlap ratios
- normalized structural similarity

The implementation should map these features to scalar attention bias values using small MLP or linear projections. Bias computation should be cacheable per hypergraph.

### Graph, Motif, And Community Readout

- `node_emb`: final node token states
- `edge_emb`: final edge token states
- `graph_emb`: pooled node tokens concatenated with pooled edge tokens, then projected
- `motif_emb`: pooled node and edge token states over sampled motif subhypergraphs
- `community_emb`: pooled node or edge token states over detected communities

## Structure Regularization

### Recommended Version

Use incident pair consistency regularization:

```python
L_struct = mean(|| W_s z_v - z_e ||^2 for (v, e) in incidence_pairs)
```

where `(v, e)` is an incident node-edge pair and `W_s` is a learnable linear map.

This regularizer is always present during pretraining and is no longer treated as only task-specific supervision.

## Pretraining Tasks

### 1. Node Pattern Prediction

- build offline node structural descriptors
- cluster them into pseudo labels
- classify node tokens into pseudo labels

### 2. Hyperedge Prediction

- use real hyperedges as positives
- generate size-matched corrupted hyperedges as negatives
- classify hyperedge tokens

### 3. Motif Classification

- sample local subhypergraphs around seed hyperedges
- build motif-level structural signatures
- cluster them into pseudo labels
- classify pooled motif embeddings

### 4. Community Prediction

- detect communities on node-node or edge-edge structural views
- assign node-level community pseudo labels
- classify node tokens into community labels

The first upgraded version uses lightweight clustering on node-node structural views instead of a heavy external community detection dependency.

### 5. Global Contrastive Learning

- generate two augmented hypergraph views
- compute graph embeddings from pooled node and edge tokens
- optimize a graph-level contrastive loss

### 6. Cross-Hypergraph Prototype Alignment

- gather motif, community, or graph-level embeddings across domains
- cluster them into cross-domain prototypes
- predict prototype IDs from the corresponding embeddings

## Total Loss

The pretraining objective is:

```python
total = (
    lambda_struct * L_struct
    + lambda_node * L_node
    + lambda_edge * L_edge
    + lambda_motif * L_motif
    + lambda_comm * L_comm
    + lambda_global * L_global
    + lambda_cross * L_cross
)
```

Recommended initial weights:

- `struct: 1.0`
- `node: 1.0`
- `edge: 1.0`
- `motif: 1.0`
- `community: 0.5`
- `global: 0.5`
- `cross: 1.0`

## Module Changes

### Files To Rewrite

- `models/backbone.py`
- `models/encoder.py`
- `models/pretext_tasks.py`
- `trainers/pretrain_trainer.py`
- `configs/pretrain.yaml`

### Files To Extend

- `models/heads.py`
- `utils/sampling.py`
- `utils/clustering.py`
- `utils/hypergraph.py`
- `trainers/finetune_trainer.py`
- `configs/finetune_node.yaml`
- `configs/finetune_edge.yaml`
- `configs/finetune_graph.yaml`
- `README.md`

### Legacy Cleanup

After the migration, remove or replace:

- HGNN convolution-specific logic
- edge embedding paths that only exist to compensate for missing edge tokens
- stale README references to the HGNN-only version
- unused imports and helpers left behind by the migration

## Public Interface

The encoder interface remains:

```python
node_emb, edge_emb, graph_emb, aux = encoder(hg, x)
```

The upgraded `aux` must expose:

- `motif_emb`
- `community_emb`
- `motifs`
- `communities`
- `incidence`
- `node_bias`
- `edge_bias`
- `node_pe`
- `edge_pe`

The pretraining loss dictionary must expose:

```python
loss_dict = {
    "struct": ...,
    "node": ...,
    "edge": ...,
    "motif": ...,
    "community": ...,
    "global": ...,
    "cross": ...,
    "total": ...,
}
```

## Training Pipeline

### Optimizer

- use `AdamW`
- keep CPU-safe defaults for the shipped config
- allow warmup-related config fields even if the first runnable version uses a simplified schedule

### Efficiency

The implementation must keep the demo pipeline runnable by:

- limiting hidden size and layer count in default configs
- caching structural features when possible
- keeping the first shipped attention implementation dense but bounded to small synthetic graphs

### Downstream Use

The downstream interface remains the same, but `graph_emb` now comes from pooled node and edge tokens instead of node-only pooling.

## Sanity Checks

The upgraded version must pass:

- encoder forward on synthetic hypergraphs
- non-NaN values for all seven loss terms
- successful pretrain CLI run
- successful transfer CLI runs for node, edge, and graph tasks
- successful ablation CLI run for one dropped task
- zero Python diagnostics after refactor

## Constraints

- keep comments concise and in English
- avoid unnecessary cloning or deep nesting
- remove dead code introduced by the old HGNN-specific path when it no longer serves the new architecture
