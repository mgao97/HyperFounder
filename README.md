# HyperFounder

Plugin-style hypergraph pretraining research scaffold based on a Hypergraph Transformer / HyperGT-style backbone and EasyHypergraph-native datasets.

## Implemented Scope

- dual-token Hypergraph Transformer backbone
- incidence-based node and hyperedge positional encoding
- node-node and edge-edge structural attention bias
- backbone-level node-hyperedge structure regularization
- 6-task pretraining loop:
  - node pattern prediction
  - hyperedge prediction
  - motif classification
  - community prediction
  - global contrastive learning
  - cross-hypergraph prototype alignment
- cross-domain transfer scripts for node, edge, and graph tasks
- single-task ablation entry point
- EasyHypergraph-native dataset registry with real dataset loading
- domain grouping across citation, authorship, commerce, political, education, and review datasets
- hyperedge-centered subhypergraph/minibatch sampling for large-graph pretraining
- optional offline subhypergraph pool for very large datasets such as `walmart_trips`
- explicit cross-domain structural pretraining with one shared HyperGT encoder

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python scripts/run_pretrain.py --config configs/pretrain.yaml
python scripts/run_pretrain.py --config configs/pretrain_large_graph.yaml
python scripts/run_transfer.py --config configs/finetune_node.yaml --heldout_domain c
python scripts/run_transfer.py --config configs/finetune_node.yaml --heldout_domain a
python scripts/run_transfer.py --config configs/finetune_node.yaml --heldout_domain e
python scripts/run_ablation.py --config configs/pretrain.yaml --drop_task community
```

## Notes

- The active data path now uses EasyHypergraph-native dataset loaders via `Python-EasyGraph`.
- The current backbone skips explicit node-hyperedge cross-attention and instead couples the two token streams through structural encoding and `L_struct`.
- Large graphs are trained through hyperedge-centered seeded subhypergraph minibatches instead of whole-graph dense attention.
- The default `pretrain.yaml` now performs cross-domain pretraining over six domains with one shared HyperGT encoder.
- The pretraining code path is:
  - `scripts/run_pretrain.py`
  - `trainers/pretrain_trainer.py`
  - `utils/minibatch_sampling.py`
- The downstream evaluation code path is:
  - `scripts/run_transfer.py`
  - `trainers/finetune_trainer.py`
- The current node-level transfer config focuses on moderate-size held-out domains: `citation`, `authorship`, `political`, and `education`.
- Edge-level and graph-level transfer are rejected when official labels are unavailable in the selected datasets.
