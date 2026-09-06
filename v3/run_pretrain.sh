#!/usr/bin/env bash
# Pre-train HyperGFSE on a multi-domain pool, then evaluate on all 8 datasets.
# Usage: bash scripts/run_pretrain.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
cd "$ROOT"

DATASETS="news20 ca_cora cc_cora cc_citeseer dblp4k_paper dblp4k_term dblp4k_conf imdb_aw"

echo "=== [1/2] Pre-training on 6-domain pool (dblp4k_term/conf held out as unseen views) ==="
$PY main.py \
  --pretrain_datasets news20 ca_cora cc_cora cc_citeseer dblp4k_paper imdb_aw \
  --epochs "${EPOCHS:-50}" \
  --max_nodes "${MAX_NODES:-3000}" \
  --output_dir outputs/hypergfse_pretrain \
  --device "${DEVICE:-cpu}"

echo "=== [2/2] Linear-probe evaluation on ALL 8 datasets ==="
$PY evaluate.py \
  --checkpoint outputs/hypergfse_pretrain/hypergfse_encoder.pt \
  --eval_datasets all \
  --folds "${FOLDS:-10}" \
  --output outputs/hypergfse_eval.csv \
  --random_baseline

echo "Done. Results -> outputs/hypergfse_eval.csv"
