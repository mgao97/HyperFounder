#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/GSK/mgao/HyperFounder"
LOG_DIR="$ROOT/logs"
OUT_DIR="$ROOT/outputs_v2/p0_frozen_vs_finetune"
CURVE_DIR="$OUT_DIR/curves"
SLS_PY="/home/user/.conda/envs/sls/bin/python"
GPU_ID="${GPU_ID:-5}"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$CURVE_DIR"

cd "$ROOT"
rm -f "$OUT_DIR/finetune_scratch.csv"
find "$CURVE_DIR" -maxdepth 1 -type f -name 'finetune_scratch_*' -delete

CUDA_VISIBLE_DEVICES="$GPU_ID" nohup bash -lc '
export PYTHONNOUSERSITE=1
export PYTHONPATH="/home/user/GSK/mgao/HyperFounder"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

"/home/user/.conda/envs/sls/bin/python" v2/scripts/run_nodecls_finetune.py \
  --scratch_no_pretrain \
  --pretrain_ckpt "" \
  --seeds 1,2,3 \
  --device cuda:0 \
  --out_csv outputs_v2/p0_frozen_vs_finetune/finetune_scratch.csv \
  --variant_tag finetune_scratch
' > "$LOG_DIR/p0_finetune_scratch.nohup.log" 2>&1 &

echo $! > "$LOG_DIR/p0_finetune_scratch.pid"
echo "[launcher] started finetune_scratch on GPU $GPU_ID"
echo "[launcher] pid=$(cat "$LOG_DIR/p0_finetune_scratch.pid")"
echo "[launcher] log=$LOG_DIR/p0_finetune_scratch.nohup.log"
