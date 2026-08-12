#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="${1:-configs/finetune_neg_sam_v2.yaml}"
HELDOUT_DOMAIN="${2:-citation}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
GPU_ID="${3:-1}"  # Default to GPU 1

OUTPUT_DIR="${4:-outputs_neg_sam_v2}"

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/pids" "$OUTPUT_DIR/results"

timestamp="$(date +%Y%m%d_%H%M%S)"
config_name="$(basename "$CONFIG_PATH" .yaml)"
log_file="$OUTPUT_DIR/logs/finetune_${config_name}_${HELDOUT_DOMAIN}_${timestamp}.log"
pid_file="$OUTPUT_DIR/pids/finetune_${config_name}_${HELDOUT_DOMAIN}_${timestamp}.pid"

cmd=( "$PYTHON_BIN" -u scripts/run_transfer.py --config "$CONFIG_PATH" --heldout_domain "$HELDOUT_DOMAIN" )

nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" "${cmd[@]}" > "$log_file" 2>&1 &
pid=$!

sleep 2

if ! ps -p "$pid" > /dev/null 2>&1; then
  echo "Finetune failed to start. Inspect log: $PROJECT_ROOT/$log_file" >&2
  if [[ -f "$log_file" ]]; then
    tail -n 40 "$log_file" >&2 || true
  fi
  exit 1
fi

echo "$pid" > "$pid_file"

echo "=============================================="
echo "Started finetune evaluation"
echo "=============================================="
echo "PID: $pid"
echo "Config: $CONFIG_PATH"
echo "Heldout Domain: $HELDOUT_DOMAIN"
echo "Log: $PROJECT_ROOT/$log_file"
echo ""
echo "Stop: kill $pid"
echo "Monitor: tail -f $log_file"
