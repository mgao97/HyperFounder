#!/usr/bin/env bash
# P1-3 trade-off grid
#   rows = {none, CCA, HCA, HOR, CCA+HCA, CCA+HOR, HCA+HOR, full}
#   cols = pretext loss / downstream Δ(frozen) / downstream Δ(finetune)
#
# Current default:
#   - pretrain seeds = 42 (as required by plan; can override)
#   - downstream eval seeds = 1,2,3
#   - runs pretrain -> frozen probe -> finetune per row
#
# Host terminal only.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

mkdir -p logs outputs_v2/p1_tradeoff_grid

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"   # 强制覆盖，防 shell-inherited expandable_segments 污染

GPU_ID="${GPU_ID:-7}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
PRETRAIN_SEED="${PRETRAIN_SEED:-42}"
EVAL_SEEDS="${EVAL_SEEDS:-1,2,3}"

# ---- 硬编码 sls 解释器（torch 1.13+cu117，已在 P0-1/2 上验证 GPU 可用）----
# 注意：command -v python / conda activate sls 在 TRAE sandbox nohup 下失败，
# 导致回落到 grag env (torch 2.x without CUDA drivers in sandbox) 并强制 CPU。
# 直接指向绝对路径避免任何 PATH / conda init 干扰。
SLS_PY="/home/user/.conda/envs/sls/bin/python"
if [ ! -x "$SLS_PY" ]; then
  echo "[p1-grid] FATAL: sls env python not found at $SLS_PY" >&2
  exit 2
fi
# 直接 probe CUDA 可见性（不走 command -v python 的 PATH）
CUDA_OK="no"
if "$SLS_PY" -c "import torch; import sys; sys.exit(0 if (torch.cuda.is_available() and torch.cuda.device_count()>0) else 1)" 2>/tmp/p1grid_cuda_probe.$$.log; then
  CUDA_OK="yes"
fi
if [ "$CUDA_OK" = "no" ]; then
  echo "[p1-grid] WARN: sls torch reports CUDA unavailable (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES).  Proceeding anyway — sandbox often hides NVML; real host GPU should be OK."
  cat /tmp/p1grid_cuda_probe.$$.log 2>/dev/null || true
fi
PY_BIN="$SLS_PY"

PRETRAIN="$PY_BIN v2/scripts/run_pretrain_v2.py"
FROZEN="$PY_BIN v2/scripts/run_w1w2_lodo_linearprobe.py"
FINETUNE="$PY_BIN v2/scripts/run_nodecls_finetune.py"

echo "[p1-grid] $(date +%F_%T) gpu=$GPU_ID pretrain_seed=$PRETRAIN_SEED eval_seeds=$EVAL_SEEDS cuda_ok=$CUDA_OK py=$PY_BIN"

run_row() {
  local tag="$1"; shift
  local extra=( "$@" )
  local out="$ROOT/outputs_v2/p1_tradeoff_grid/$tag"
  local ckpt="$out/checkpoints/pretrain_best_v2.pt"
  mkdir -p "$out"

  echo ""
  echo "==================================================================="
  echo "[p1-grid] row=$tag pretrain @ $(date +%F_%T)"
  echo "          out=$out"
  echo "          extra=(${extra[*]:-})"
  echo "==================================================================="
  # shellcheck disable=SC2086
  $PRETRAIN \
    --config v2/configs/pretrain_v2.yaml \
    --seed "$PRETRAIN_SEED" \
    --output_dir "$out" \
    "${extra[@]}" \
    2>&1 | tee "logs/p1_grid_train_${tag}.log"

  echo "[p1-grid] row=$tag frozen probe @ $(date +%F_%T)"
  $FROZEN \
    --pretrain_ckpt "$ckpt" \
    --seeds "$EVAL_SEEDS" \
    --device cuda:0 \
    --out_csv "$out/frozen.csv" \
    --variant_tag "$tag" \
    "${extra[@]}" \
    2>&1 | tee "logs/p1_grid_frozen_${tag}.log"

  echo "[p1-grid] row=$tag finetune @ $(date +%F_%T)"
  $FINETUNE \
    --pretrain_ckpt "$ckpt" \
    --seeds "$EVAL_SEEDS" \
    --device cuda:0 \
    --out_csv "$out/finetune.csv" \
    --variant_tag "$tag" \
    "${extra[@]}" \
    2>&1 | tee "logs/p1_grid_finetune_${tag}.log"
}

# none = no CCA, no HCA, no HOR
run_row none --ablate_cca_card --ablate_cca_film --ablate_cca_tau --ablate_hca_full --use_hor false
# CCA only
run_row CCA --ablate_hca_full --use_hor false
# HCA only
run_row HCA --ablate_cca_card --ablate_cca_film --ablate_cca_tau --use_hor false
# HOR only
run_row HOR --ablate_cca_card --ablate_cca_film --ablate_cca_tau --ablate_hca_full --use_hor true
# CCA + HCA
run_row CCA_HCA --use_hor false
# CCA + HOR
run_row CCA_HOR --ablate_hca_full --use_hor true
# HCA + HOR
run_row HCA_HOR --ablate_cca_card --ablate_cca_film --ablate_cca_tau --use_hor true
# full
run_row full --use_hor true

echo "[p1-grid] DONE @ $(date +%F_%T)"
