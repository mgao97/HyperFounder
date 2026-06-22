# LLM Text-Feature Downstream Task (Design Notes)

## Goal

Add an experimental evaluation track that verifies the proposed HyperGT-style structural pretraining can work on top of existing LLM-extracted text features, and then solve downstream tasks on **text-attributed hypergraph datasets**.

## Current Status (What The Codebase Supports Today)

- The current EasyHypergraph dataset adapter only consumes numeric fields: `edge_list`, `labels`, optional `features`, optional split masks. See [load_easyhypergraph_sample](file:///Users/santa/Desktop/HyperFounder/utils/easyhypergraph_datasets.py#L64-L105).
- The registered datasets in [dataset_registry.py](file:///Users/santa/Desktop/HyperFounder/utils/dataset_registry.py#L27-L40) do not expose any text-like keys (`text/title/abstract/review/...`) through `dataset.content` / `dataset._content`.
  - Verified by [inspect_dataset_text_fields.py](file:///Users/santa/Desktop/HyperFounder/scripts/inspect_dataset_text_fields.py).
- OpenAI is not configured in the current environment, so in-repo “call an LLM API to embed text” cannot run without configuration.

## What Needs To Change To Support Text-Attributed Hypergraphs

At minimum, one of the following must be true:

1. The dataset loader exposes raw text (per node / per hyperedge / per graph), and we integrate an embedding step; or
2. Raw text is embedded offline (existing work), and this repo only loads the resulting embeddings as node features.

Given the repo currently has no `transformers`/`sentence-transformers` dependency and OpenAI is unconfigured, the lowest-risk path is (2).

## Recommended Experimental Protocol (Using Precomputed LLM Embeddings)

### Data Assumption

For each text-attributed dataset:

- Hypergraph structure: `edge_list` / incidence.
- Labels: node-level labels for evaluation (node classification), plus optional train/val/test masks.
- Text embeddings: a numeric matrix `X_text ∈ R^{N×d}` produced externally (e.g., by an LLM encoder).

### Downstream Task

- **Node classification transfer** (already implemented): [FinetuneTrainer](file:///Users/santa/Desktop/HyperFounder/trainers/finetune_trainer.py).

### Evaluation Comparisons

Run at least:

- Baseline A: downstream classifier on `X_text` only (no structural pretraining; encoder randomly initialized or bypassed).
- Baseline B: HyperGT encoder trained without structural pretraining (disable all pretext losses or run very short pretraining).
- Proposed: HyperGT structural pretraining + downstream finetuning, using the same `X_text`.

### What “Success” Means

- Proposed method improves node accuracy vs Baseline A/B under the same text embeddings, especially in held-out-domain transfer.

## Open Questions (Need Your Decision)

1. Do you want this repo to **embed raw text inside the pipeline** (requires adding dependencies and/or configuring an API key), or **only load precomputed embeddings** produced externally?
2. Which specific text-attributed hypergraph datasets are required for the paper/experiments? (They are not among the currently registered EasyHypergraph datasets.)

