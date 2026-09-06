#!/usr/bin/env bash
# S0-1 main contribution check:
#   strict full ckpt (seed42) vs scratch encoder
#   under both frozen and finetune W1/W2 LODO protocols.
#
# Outputs:
#   outputs_v2/s0_main_contribution/{frozen_full,frozen_scratch,finetune_full,finetune_scratch}.csv
#   logs/s0_main_contribution_summary.md
#
# Usage:
#   cd /home/user/GSK/mgao/HyperFounder
#   GPU_ID=7 bash v2/scripts/run_s0_main_contribution_check.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

OUT_DIR="$ROOT/outputs_v2/s0_main_contribution"
LOG_DIR="$ROOT/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

GPU_ID="${GPU_ID:-7}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
SEEDS="${SEEDS:-1,2,3}"
FULL_CKPT="${FULL_CKPT:-outputs_v2/ablations_seed42/w3_full/checkpoints/pretrain_best_v2.pt}"
SLS_PY="/home/user/.conda/envs/sls/bin/python"

if [ ! -x "$SLS_PY" ]; then
  echo "[s0] FATAL: python not found: $SLS_PY" >&2
  exit 2
fi

echo "[s0] $(date +%F_%T) gpu=$GPU_ID seeds=$SEEDS py=$SLS_PY"

"$SLS_PY" v2/scripts/run_w1w2_lodo_linearprobe.py \
  --pretrain_ckpt "$FULL_CKPT" \
  --seeds "$SEEDS" \
  --device cuda:0 \
  --out_csv outputs_v2/s0_main_contribution/frozen_full.csv \
  --variant_tag frozen_full \
  2>&1 | tee "$LOG_DIR/s0_frozen_full.log"

"$SLS_PY" v2/scripts/run_w1w2_lodo_linearprobe.py \
  --scratch_no_pretrain \
  --pretrain_ckpt "" \
  --seeds "$SEEDS" \
  --device cuda:0 \
  --out_csv outputs_v2/s0_main_contribution/frozen_scratch.csv \
  --variant_tag frozen_scratch \
  2>&1 | tee "$LOG_DIR/s0_frozen_scratch.log"

"$SLS_PY" v2/scripts/run_nodecls_finetune.py \
  --pretrain_ckpt "$FULL_CKPT" \
  --seeds "$SEEDS" \
  --device cuda:0 \
  --out_csv outputs_v2/s0_main_contribution/finetune_full.csv \
  --variant_tag finetune_full \
  2>&1 | tee "$LOG_DIR/s0_finetune_full.log"

"$SLS_PY" v2/scripts/run_nodecls_finetune.py \
  --scratch_no_pretrain \
  --pretrain_ckpt "" \
  --seeds "$SEEDS" \
  --device cuda:0 \
  --out_csv outputs_v2/s0_main_contribution/finetune_scratch.csv \
  --variant_tag finetune_scratch \
  2>&1 | tee "$LOG_DIR/s0_finetune_scratch.log"

"$SLS_PY" v2/scripts/summarize_s0_main_contribution.py \
  --in_dir outputs_v2/s0_main_contribution \
  --out_md logs/s0_main_contribution_summary.md

echo "[s0] DONE @ $(date +%F_%T)"
