#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# scripts/nohup_linear_probe_neg_sam.sh
#
# Background runner for the linear-probe evaluation of a pretrained
# checkpoint. Trains a 1-layer MLP on top of the FROZEN encoder and
# reports mean ± std accuracy / macro-F1 over multiple seeds.
#
# This is the post-pretrain sanity check (Phase 5 in
# scripts/full_pretrain_pipeline.sh). Run AFTER the pretrain finishes
# (or, for partial pretrain, after it has produced
# outputs_neg_sam_v2/checkpoints/pretrain_best_neg_sam.pt).
#
# Usage:
#   bash scripts/nohup_linear_probe_neg_sam.sh
#   bash scripts/nohup_linear_probe_neg_sam.sh outputs_neg_sam_v2/checkpoints/pretrain_best_neg_sam.pt
#   SEEDS="7 13 42" DATASETS="cora_cc cooking_200" \
#       bash scripts/nohup_linear_probe_neg_sam.sh
#
# Env vars:
#   DEVICE                  cpu|cuda                       (default: cuda)
#   SEEDS                   space-sep ints                 (default: 7 13 42)
#   DATASETS                space-sep names                (default: cora_cc cooking_200 gowalla)
#   EPOCHS                  int                            (default: 100)
#   PATIENCE                int                            (default: 20)
#   PRETRAIN_CONFIG          yaml path                      (default: configs/pretrain_neg_sam_v2.yaml)
# -----------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PRETRAINED_CKPT="${1:-outputs_neg_sam_v2/checkpoints/pretrain_best_neg_sam.pt}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-7 13 42}"
DATASETS="${DATASETS:-cora_cc cooking_200 gowalla}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-20}"
PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-configs/pretrain_neg_sam_v2.yaml}"

if [[ ! -f "$PRETRAINED_CKPT" ]]; then
  echo "Pretrained checkpoint not found: $PRETRAINED_CKPT" >&2
  echo "Run pretrain first: bash scripts/nohup_pretrain_neg_sam_hedg.sh" >&2
  exit 1
fi
if [[ ! -f "$PRETRAIN_CONFIG" ]]; then
  echo "PRETRAIN_CONFIG not found: $PRETRAIN_CONFIG" >&2
  exit 1
fi

mkdir -p outputs_neg_sam_v2/logs outputs_neg_sam_v2/pids

timestamp="$(date +%Y%m%d_%H%M%S)"
ckpt_name="$(basename "$PRETRAINED_CKPT" .pt)"
log_file="outputs_neg_sam_v2/logs/linear_probe_${ckpt_name}_${timestamp}.log"
pid_file="outputs_neg_sam_v2/pids/linear_probe_${ckpt_name}_${timestamp}.pid"
latest_pid_file="outputs_neg_sam_v2/pids/linear_probe_latest.pid"
latest_meta_file="outputs_neg_sam_v2/pids/linear_probe_latest.meta"

cmd=(
  python3 scripts/linear_probe_neg_sam.py
    --device "$DEVICE"
    --pretrained "$PRETRAINED_CKPT"
    --config "$PRETRAIN_CONFIG"
    --datasets $DATASETS
    --seeds $SEEDS
    --epochs "$EPOCHS"
    --patience "$PATIENCE"
    --output outputs_neg_sam_v2/results/linear_probe_neg_sam.json
)

nohup "${cmd[@]}" > "$log_file" 2>&1 &
pid=$!

sleep 3

if ! ps -p "$pid" > /dev/null 2>&1; then
  echo "Linear probe failed to start. Inspect log: $PROJECT_ROOT/$log_file" >&2
  if [[ -f "$log_file" ]]; then
    tail -n 80 "$log_file" >&2 || true
  fi
  exit 1
fi

echo "$pid" > "$pid_file"
echo "$pid" > "$latest_pid_file"
cat > "$latest_meta_file" <<EOF
PID=$pid
DEVICE=$DEVICE
PRETRAINED_CKPT=$PROJECT_ROOT/$PRETRAINED_CKPT
PRETRAIN_CONFIG=$PROJECT_ROOT/$PRETRAIN_CONFIG
DATASETS=$DATASETS
SEEDS=$SEEDS
LOG=$PROJECT_ROOT/$log_file
STARTED_AT=$timestamp
EOF

echo "=============================================="
echo "Started linear probe"
echo "=============================================="
echo "PID:           $pid"
echo "Checkpoint:    $PROJECT_ROOT/$PRETRAINED_CKPT"
echo "Datasets:      $DATASETS"
echo "Seeds:         $SEEDS"
echo "Log:           $PROJECT_ROOT/$log_file"
echo "Result:        outputs_neg_sam_v2/results/linear_probe_neg_sam.json"
echo ""
echo "Monitor:"
echo "  tail -f $PROJECT_ROOT/$log_file"
echo "Stop:"
echo "  kill \$(cat $pid_file)"
