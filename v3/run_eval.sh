#!/usr/bin/env bash
# Evaluate an existing HyperGFSE checkpoint on all 8 datasets (linear probe).
# Usage: bash scripts/run_eval.sh  [path/to/hypergfse_encoder.pt]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
cd "$ROOT"

CKPT="${1:-outputs/hypergfse_pretrain/hypergfse_encoder.pt}"
OUT="${2:-outputs/hypergfse_eval.csv}"

$PY evaluate.py \
  --checkpoint "$CKPT" \
  --eval_datasets all \
  --folds "${FOLDS:-10}" \
  --output "$OUT" \
  --random_baseline

echo "Done. Results -> $OUT"
