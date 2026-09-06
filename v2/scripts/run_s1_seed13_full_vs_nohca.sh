#!/usr/bin/env bash
# S1-1: add pretrain seed=13 for just two variants: full vs no_HCA.
#
# Decision rule after run:
#   if no_HCA wins in >= 2 of {1,7,13}, demote HCA from the main configuration.
#
# Outputs:
#   outputs_v2/s1_seed13_full_vs_nohca/{full,no_HCA}/...
#   logs/s1_seed13_full_vs_nohca_summary.md
#
# Usage:
#   cd /home/user/GSK/mgao/HyperFounder
#   GPU_ID=7 bash v2/scripts/run_s1_seed13_full_vs_nohca.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

OUT_BASE="$ROOT/outputs_v2/s1_seed13_full_vs_nohca"
LOG_DIR="$ROOT/logs"
mkdir -p "$OUT_BASE" "$LOG_DIR"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

GPU_ID="${GPU_ID:-7}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
SEED="${PRETRAIN_SEED:-13}"
SLS_PY="/home/user/.conda/envs/sls/bin/python"

if [ ! -x "$SLS_PY" ]; then
  echo "[s1] FATAL: python not found: $SLS_PY" >&2
  exit 2
fi

echo "[s1] $(date +%F_%T) gpu=$GPU_ID pretrain_seed=$SEED py=$SLS_PY"

run_variant() {
  local tag="$1"; shift
  local extra=( "$@" )
  local out_dir="$OUT_BASE/$tag"
  local ckpt="$out_dir/checkpoints/pretrain_best_v2.pt"

  mkdir -p "$out_dir"
  echo ""
  echo "==================================================================="
  echo "[s1] variant=$tag pretrain @ $(date +%F_%T)"
  echo "      out=$out_dir"
  echo "      extra=(${extra[*]:-})"
  echo "==================================================================="

  "$SLS_PY" v2/scripts/run_pretrain_v2.py \
    --config v2/configs/pretrain_v2.yaml \
    --seed "$SEED" \
    --output_dir "$out_dir" \
    "${extra[@]}" \
    2>&1 | tee "$LOG_DIR/s1_seed13_train_${tag}.log"

  echo "[s1] variant=$tag probe @ $(date +%F_%T)"
  "$SLS_PY" v2/scripts/run_w1w2_lodo_linearprobe.py \
    --pretrain_ckpt "$ckpt" \
    --seeds 1,2,3 \
    --device cuda:0 \
    --out_csv "outputs_v2/s1_seed13_full_vs_nohca/${tag}/lodo_probe.csv" \
    --variant_tag "s1_seed13_${tag}" \
    "${extra[@]}" \
    2>&1 | tee "$LOG_DIR/s1_seed13_probe_${tag}.log"
}

run_variant full
run_variant no_HCA --ablate_hca_full

"$SLS_PY" v2/scripts/summarize_s1_seed13_full_vs_nohca.py \
  --seed13_dir outputs_v2/s1_seed13_full_vs_nohca \
  --out_md logs/s1_seed13_full_vs_nohca_summary.md

echo "[s1] DONE @ $(date +%F_%T)"
