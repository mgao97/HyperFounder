#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="${1:-configs/pretrain_neg_sam_v2.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"  # Using GPU mode
GPU_ID="${CUDA_VISIBLE_DEVICES:-2}"  # Use GPU 2

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p outputs_neg_sam_v2/logs outputs_neg_sam_v2/pids outputs_neg_sam_v2/checkpoints outputs_neg_sam_v2/results

timestamp="$(date +%Y%m%d_%H%M%S)"
config_name="$(basename "$CONFIG_PATH" .yaml)"
log_file="outputs_neg_sam_v2/logs/${config_name}_${timestamp}.log"
pid_file="outputs_neg_sam_v2/pids/${config_name}_${timestamp}.pid"
latest_pid_file="outputs_neg_sam_v2/pids/pretrain_latest.pid"
latest_meta_file="outputs_neg_sam_v2/pids/pretrain_latest.meta"

if [[ "$DEVICE" == "cuda" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

cmd=( "$PYTHON_BIN" -u scripts/run_pretrain_neg_sam.py --config "$CONFIG_PATH" --device "$DEVICE" )

nohup "${cmd[@]}" > "$log_file" 2>&1 &
pid=$!

sleep 2

if ! ps -p "$pid" > /dev/null 2>&1; then
  echo "Pretraining failed to start. Inspect log: $PROJECT_ROOT/$log_file" >&2
  if [[ -f "$log_file" ]]; then
    tail -n 40 "$log_file" >&2 || true
  fi
  exit 1
fi

echo "$pid" > "$pid_file"
echo "$pid" > "$latest_pid_file"
cat > "$latest_meta_file" <<EOF
PID=$pid
DEVICE=$DEVICE
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
CONFIG=$PROJECT_ROOT/$CONFIG_PATH
LOG=$PROJECT_ROOT/$log_file
STARTED_AT=$timestamp
EOF

echo "=============================================="
echo "Started pretraining (neg_sam_v2 optimized)"
echo "Testing GPU 2"
echo "=============================================="
echo "PID: $pid"
echo "Device: $DEVICE"
echo "CUDA_VISIBLE_DEVICES: $GPU_ID"
echo "Config: $CONFIG_PATH"
echo "Log: $PROJECT_ROOT/$log_file"
echo "Output: outputs_neg_sam_v2/"
echo ""
echo "Key optimizations:"
echo "  - num_neg_per_pos: 4 -> 2"
echo "  - membership_contrast weight: 0.5 -> 0.2"
echo "  - min_nodes_for_node_contrastive: 16 -> 8"
echo "  - early_stopping patience: 50 -> 80"
echo "  - contrastive_temperature: 0.07 -> 0.1"
echo ""
echo "Stop: bash scripts/stop_pretrain_neg_sam_v2.sh"
echo "Status: bash scripts/status_pretrain_neg_sam_v2.sh"
