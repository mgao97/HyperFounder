# Hypergraph Pretraining Design

## Objective

Implement a first-version research codebase under `/Users/santa/Desktop/HyperFounder` for a plugin-style unified hypergraph structural encoder built on top of an HGNN backbone. The implementation must follow the requirements extracted from `实现描述.pdf` and support:

- A minimum closed loop of 5 pretraining tasks:
  - node pattern prediction
  - hyperedge prediction
  - motif classification
  - global contrastive learning
  - cross-hypergraph prototype alignment
- Cross-domain pretraining and transfer evaluation
- Ablation by dropping a single pretraining task
- Reusable encoder outputs for downstream node-, edge-, and graph-level tasks

The first version intentionally excludes the community-level task to keep scope manageable and aligned with the approved minimum implementation target.

## Scope

### In Scope

- HGNN-style backbone with residual support
- Unified encoder producing node, edge, graph, and auxiliary motif outputs
- Modular heads for the five pretraining tasks
- Pretraining trainer and transfer/finetune trainer
- Config-driven experiment entry scripts
- Logging of losses, seeds, held-out domain metrics, and ablation CSV outputs
- DHG-compatible data interfaces with lightweight adapters

### Out of Scope

- Community-level pseudo-supervision
- Full benchmark dataset downloading automation for every external source
- Paper-ready plotting and figure generation
- Exhaustive backbone variants beyond the interfaces needed for future extension

## Architecture

### Core Design

The implementation uses a plugin-style encoder rather than a tightly coupled end-to-end single-task model. The encoder is reusable across pretraining and downstream transfer.

Core flow:

1. Build node embeddings with an HGNN backbone.
2. Derive hyperedge embeddings by pooling member node embeddings.
3. Derive graph embeddings by graph-level readout on node embeddings.
4. Sample motif/subhypergraph units and build motif embeddings from pooled node and edge representations.
5. Feed these representations into task-specific pretraining heads.

### Backbone

The first version uses a two-layer HGNN-style encoder with:

- hidden dimension configurable via YAML
- ReLU activation
- dropout in the `0.3` to `0.5` range
- shallow residual connection to reduce oversmoothing

If `dhg` is available, the code should support DHG objects directly. If not, the code should remain import-safe and structured around DHG-compatible assumptions so the environment can be completed later without rewriting the project.

### Readout Strategy

- Node embedding: backbone output per node
- Hyperedge embedding: `mean` + `max` pooling over member node embeddings, followed by projection
- Graph embedding: `mean` pooling over node embeddings, with an option to extend to attention pooling later
- Motif embedding: pooled node and edge embeddings from sampled local subhypergraphs

## Modules

### File Layout

The project should contain:

- `configs/pretrain.yaml`
- `configs/finetune_node.yaml`
- `configs/finetune_edge.yaml`
- `configs/finetune_graph.yaml`
- `models/backbone.py`
- `models/encoder.py`
- `models/heads.py`
- `models/pretext_tasks.py`
- `trainers/pretrain_trainer.py`
- `trainers/finetune_trainer.py`
- `utils/sampling.py`
- `utils/clustering.py`
- `utils/eval.py`
- `utils/metrics.py`
- `scripts/run_pretrain.py`
- `scripts/run_transfer.py`
- `scripts/run_ablation.py`

### Module Responsibilities

- `backbone.py`: HGNN-style node encoder and backbone factory
- `encoder.py`: unified forward path and multi-granularity readouts
- `heads.py`: reusable MLP heads for node, edge, motif, graph, and prototype tasks
- `pretext_tasks.py`: task builders and loss computation
- `pretrain_trainer.py`: multi-task pretraining loop, logging, checkpointing
- `finetune_trainer.py`: transfer evaluation for node, edge, and graph tasks
- `sampling.py`: edge negative sampling, motif sampling, augmentation helpers
- `clustering.py`: offline pseudo-label generation and prototype refresh
- `eval.py`: seed aggregation, held-out domain evaluation, CSV writing
- `metrics.py`: classification and ranking metrics

## Public Interfaces

### Encoder Interface

The encoder contract is:

```python
node_emb, edge_emb, graph_emb, aux = encoder(hg, x)
```

Expected outputs:

- `hg`: hypergraph object, preferably DHG-compatible
- `x`: node features with shape `[num_nodes, in_dim]`
- `node_emb`: `[num_nodes, hidden_dim]`
- `edge_emb`: `[num_edges, hidden_dim]`
- `graph_emb`: `[hidden_dim]` or `[1, hidden_dim]`
- `aux`: dictionary for motif embeddings, motif assignments, masks, and metadata

### Pretraining Output

Each pretraining step returns:

```python
loss_dict = {
    "node": ...,
    "edge": ...,
    "motif": ...,
    "global": ...,
    "cross": ...,
    "total": ...,
}
```

## Pretraining Tasks

### 1. Node Pattern Prediction

Purpose:

- capture node degree, local structural role, and higher-order neighborhood patterns

Implementation:

- compute offline structural descriptors per node
- cluster descriptors into pseudo-labels using KMeans
- predict cluster IDs from node embeddings with a classification head

Required descriptors:

- node degree
- incident hyperedge size mean
- incident hyperedge size max
- 2-hop reachable node count
- overlap-based neighborhood statistics

### 2. Hyperedge Prediction

Purpose:

- model hyperedge co-occurrence and membership consistency

Implementation:

- positive samples are real hyperedges
- negative samples are generated with size-matched node replacement
- classify whether a candidate hyperedge is real

Constraint:

- negative sampling must preserve hyperedge size distribution

### 3. Motif Classification

Purpose:

- learn local incidence patterns and overlapping higher-order substructures

Implementation:

- sample motif/subhypergraph neighborhoods around seed hyperedges
- build structural signatures
- cluster signatures into motif pseudo-labels
- predict motif type from motif embeddings

Required budget:

- fixed motif sampling budget per graph per epoch, default `500`

### 4. Global Contrastive Learning

Purpose:

- improve graph-level robustness and topology-aware representation learning

Implementation:

- create two augmented views of the same hypergraph
- use graph-level InfoNCE-style contrastive learning

Required augmentations:

- hyperedge dropout
- node feature masking

### 5. Cross-Hypergraph Prototype Alignment

Purpose:

- encourage domain-invariant structural representations across datasets

Implementation:

- sample motif/community-like local units from all domains
- cluster structural summaries into prototypes
- predict or contrast against prototype assignments from motif/subhypergraph embeddings

Constraint:

- prototype centers are refreshed every `N` epochs, configurable in YAML

## Data Assumptions

### Domain Organization

The code should support domain buckets such as:

- citation/academic
- content/document
- recommendation/interaction
- biological/other

Each domain entry should provide:

- dataset name
- task type
- features
- labels if available
- split metadata if available

### Loader Strategy

The project should expose a simple dataset adapter layer so training code depends on a normalized batch format rather than dataset-specific branching.

The implementation should prefer DHG-compatible hypergraph objects but must degrade gracefully if some datasets are not available locally.

## Training Protocol

### Pretraining

- multi-domain sampling within each epoch
- Adam optimizer
- learning rate default `1e-3`
- weight decay default `1e-5`
- configurable epochs, default in the `100` to `300` range
- optional early stopping on validation or pretext stabilization

### Transfer Evaluation

Support these modes:

- `scratch`: no pretraining
- `single-domain pretrain`
- `multi-domain pretrain`

Support these transfer settings:

- leave-one-domain-out
- few-shot finetuning
- full-shot finetuning
- linear probe
- full finetune

### Seeds

All experiments must run with exactly `3` seeds by default, and summaries must report mean and standard deviation.

## CLI Requirements

The implementation must support:

```bash
python scripts/run_pretrain.py --config configs/pretrain.yaml
python scripts/run_transfer.py --config configs/finetune_node.yaml --heldout_domain c
python scripts/run_transfer.py --config configs/finetune_edge.yaml --heldout_domain r
python scripts/run_ablation.py --config configs/pretrain.yaml --drop_task motif
```

`heldout_domain` should map to domain aliases defined in configuration.

## Logging And Outputs

The implementation must save:

- per-task loss curves
- per-domain pretraining sample counts
- downstream averages and standard deviations across `3` seeds
- held-out domain transfer metrics
- ablation results as CSV

Recommended output layout:

- `outputs/checkpoints/`
- `outputs/logs/`
- `outputs/results/`

## Failure Cases

The implementation should explicitly guard against:

- empty hyperedge sets
- motifs with too few nodes or edges
- prototype clusters with too few assigned samples
- mismatch between feature rows and node count
- unavailable datasets or missing DHG dependency

When these occur, the code should raise clear errors or skip safely with logged warnings rather than failing silently.

## Sanity Checks

Minimum sanity checks for the first version:

- one small synthetic hypergraph forward pass
- correct tensor shapes for node, edge, graph, and motif outputs
- non-NaN total loss for each enabled task
- one ablation run dropping a single task
- one transfer command parse and config load check

## Implementation Notes

- Use early returns to avoid deep nesting.
- Keep helper functions focused and small.
- Avoid unnecessary object copying.
- Add concise English comments only where the logic is not obvious.
- Keep the implementation modular so community-level pretraining can be added later without restructuring the core encoder.
