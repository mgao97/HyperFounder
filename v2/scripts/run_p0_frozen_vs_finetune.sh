#!/usr/bin/env bash
# P0-2 runner: scratch/full × frozen/finetune
#
# Reuses:
#   - frozen: run_w1w2_lodo_linearprobe.py
#   - finetune: run_nodecls_finetune.py
#
# Default uses the seed42 full checkpoint as the pretrained full method.
# Host terminal only.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

mkdir -p logs outputs_v2/p0_frozen_vs_finetune

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

GPU_ID="${GPU_ID:-7}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
SEEDS="${SEEDS:-1,2,3}"
FULL_CKPT="${FULL_CKPT:-outputs_v2/ablations_seed42/w3_full/checkpoints/pretrain_best_v2.pt}"

if [ -f /opt/anaconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /opt/anaconda3/etc/profile.d/conda.sh
fi
if command -v conda >/dev/null 2>&1; then conda activate sls 2>/dev/null || true ; fi

SYS_PY="$(command -v python)"
SLS_OK="no"
if [ -n "$SYS_PY" ]; then
  if "$SYS_PY" -c "import torch; exit(0 if (torch.cuda.is_available() and torch.cuda.device_count()>0) else 1)" 2>/dev/null; then
    SLS_OK="yes"
  fi
fi
GRAG_PY="/home/user/.conda/envs/grag/bin/python"
if [ "$SLS_OK" = "yes" ]; then PY_BIN="$SYS_PY"; else PY_BIN="$GRAG_PY"; fi

echo "[p0-2] $(date +%F_%T) gpu=$GPU_ID sls_ok=$SLS_OK py=$PY_BIN"

# 1) frozen + pretrained full
"$PY_BIN" v2/scripts/run_w1w2_lodo_linearprobe.py \
  --pretrain_ckpt "$FULL_CKPT" \
  --seeds "$SEEDS" \
  --device cuda:0 \
  --out_csv outputs_v2/p0_frozen_vs_finetune/frozen_full.csv \
  --variant_tag frozen_full \
  2>&1 | tee logs/p0_frozen_full.log

# 2) frozen + scratch encoder
"$PY_BIN" v2/scripts/run_w1w2_lodo_linearprobe.py \
  --scratch_no_pretrain \
  --pretrain_ckpt "" \
  --seeds "$SEEDS" \
  --device cuda:0 \
  --out_csv outputs_v2/p0_frozen_vs_finetune/frozen_scratch.csv \
  --variant_tag frozen_scratch \
  2>&1 | tee logs/p0_frozen_scratch.log

# 3) finetune + pretrained full
"$PY_BIN" v2/scripts/run_nodecls_finetune.py \
  --pretrain_ckpt "$FULL_CKPT" \
  --seeds "$SEEDS" \
  --device cuda:0 \
  --out_csv outputs_v2/p0_frozen_vs_finetune/finetune_full.csv \
  --variant_tag finetune_full \
  2>&1 | tee logs/p0_finetune_full.log

# 4) finetune + scratch encoder
"$PY_BIN" v2/scripts/run_nodecls_finetune.py \
  --scratch_no_pretrain \
  --pretrain_ckpt "" \
  --seeds "$SEEDS" \
  --device cuda:0 \
  --out_csv outputs_v2/p0_frozen_vs_finetune/finetune_scratch.csv \
  --variant_tag finetune_scratch \
  2>&1 | tee logs/p0_finetune_scratch.log

echo "[p0-2] DONE @ $(date +%F_%T)"
