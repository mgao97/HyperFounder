# Cross-Domain Hypergraph Structural Pretraining Design

## Objective

Upgrade the current HyperGT project so it is explicitly implemented as a cross-domain hypergraph structural pretraining model rather than only a multi-dataset training script.

The upgraded system must:

- pretrain one shared hypergraph structural encoder across multiple domains
- use domain-balanced sampling during pretraining
- optimize transferable structural self-supervision objectives
- support held-out-domain transfer evaluation
- record training domains and evaluation domains in outputs

## Cross-Domain Definition

The project is considered cross-domain only if all of the following are true:

- one encoder is trained jointly across multiple hypergraph domains
- each epoch samples from multiple domains in a balanced way
- pretraining tasks are structural and label-free at their core
- downstream evaluation can hold out one domain and test transfer

## Training Domains

The primary cross-domain pretraining configuration should use six domains represented by EasyHypergraph datasets that are currently available in this environment:

- `citation`
  - `cocitation_cora`
- `authorship`
  - `coauthorship_cora`
- `commerce`
  - `walmart_trips`
- `political`
  - `senate_committees`
- `education`
  - `contact_primary_school`
- `review`
  - `yelp_restaurant`

This gives the project a true multi-domain structural pretraining setup similar in spirit to a general foundational structural encoder.

## Pretraining Objectives

The current HyperGT tasks remain but are framed as universal structural pretraining tasks:

- `struct`
  - incidence consistency between node and hyperedge tokens
- `node`
  - local structural role prediction through pseudo-label clustering
- `edge`
  - hyperedge existence and completion style discrimination
- `motif`
  - higher-order local structural pattern prediction
- `community`
  - mesoscale overlap structure prediction
- `global`
  - subhypergraph-level contrastive learning
- `cross`
  - cross-domain prototype alignment

These tasks together define the structural self-supervision objective.

## Domain-Balanced Sampling

The pretraining loop must not sample graphs globally without regard to domain balance.

Instead:

- construct a domain schedule per epoch
- sample a configured number of domains per step
- sample a configured number of subhypergraphs per chosen domain
- cycle through all available domains before repeating them as much as possible

This keeps large domains such as `commerce` or `review` from dominating updates.

## Evaluation Protocol

The downstream protocol should explicitly support held-out-domain transfer.

The first evaluation-ready node classification config should cover moderate-size domains:

- `citation`
- `authorship`
- `political`
- `education`

Large domains may remain pretraining-only until scalable downstream evaluation is added.

## Result Metadata

Pretraining outputs must record:

- `cross_domain_pretraining`
- `training_domains`
- `training_datasets`
- `num_domains`
- `sampling_mode`
- `domain_sample_counts`
- `pooled_graphs`

Transfer outputs must record:

- `heldout_domain`
- `evaluated_datasets`
- `pretrain_domains` when available from the checkpoint config

## Files To Modify

- `trainers/pretrain_trainer.py`
- `trainers/finetune_trainer.py`
- `utils/minibatch_sampling.py`
- `configs/pretrain.yaml`
- `configs/finetune_node.yaml`
- `README.md`

## Optional Files To Add

- `configs/pretrain_cross_domain.yaml`
- `configs/finetune_node_cross_domain.yaml`

## Constraints

- keep the current HyperGT backbone as the shared encoder
- preserve the subhypergraph/minibatch sampling path for large graphs
- do not rely on downstream labels for pretraining
- do not silently treat single-domain training as cross-domain
