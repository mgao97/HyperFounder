#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PRETRAINED_CKPT="${1:-outputs_neg_sam_smoke/checkpoints/pretrain_best_neg_sam.pt}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
DATASETS_CSV="${DATASETS:-cora_cc cooking_200}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
SEEDS_CSV="${SEEDS:-7}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_neg_sam_smoke}"

if [[ ! -f "$PRETRAINED_CKPT" ]]; then
  echo "Pretrained checkpoint not found: $PRETRAINED_CKPT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/pids"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$OUTPUT_DIR/logs/linear_probe_${timestamp}.log"
pid_file="$OUTPUT_DIR/pids/linear_probe_${timestamp}.pid"
latest_pid_file="$OUTPUT_DIR/pids/linear_probe_latest.pid"
latest_meta_file="$OUTPUT_DIR/pids/linear_probe_latest.meta"
result_json="$OUTPUT_DIR/results/linear_probe_neg_sam.json"

cmd=(
  "$PYTHON_BIN" -u scripts/linear_probe_neg_sam.py
    --device "$DEVICE"
    --pretrained "$PRETRAINED_CKPT"
    --datasets $DATASETS_CSV
    --seeds $SEEDS_CSV
    --epochs "$EPOCHS"
    --patience "$PATIENCE"
    --output "$result_json"
)

nohup "${cmd[@]}" > "$log_file" 2>&1 &
pid=$!

sleep 2

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
DATASETS=$DATASETS_CSV
SEEDS=$SEEDS_CSV
EPOCHS=$EPOCHS
PATIENCE=$PATIENCE
LOG=$PROJECT_ROOT/$log_file
RESULT_JSON=$PROJECT_ROOT/$result_json
STARTED_AT=$timestamp
EOF

echo "=============================================="
echo "Started linear probe"
echo "=============================================="
echo "PID:                $pid"
echo "Device:             $DEVICE"
echo "Pretrained ckpt:    $PRETRAINED_CKPT"
echo "Datasets:           $DATASETS_CSV"
echo "Seeds:              $SEEDS_CSV"
echo "Epochs/Patience:    $EPOCHS / $PATIENCE"
echo "Log:                $PROJECT_ROOT/$log_file"
echo "Result JSON:        $PROJECT_ROOT/$result_json"
echo ""
echo "Tail:    tail -f $PROJECT_ROOT/$log_file"
echo "Stop:    kill \$(cat $pid_file)"
