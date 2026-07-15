#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="${1:-configs/pretrain.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p outputs/logs outputs/pids

timestamp="$(date +%Y%m%d_%H%M%S)"
config_name="$(basename "$CONFIG_PATH" .yaml)"
log_file="outputs/logs/${config_name}_pretrain_${timestamp}.log"
pid_file="outputs/pids/${config_name}_pretrain_${timestamp}.pid"
latest_pid_file="outputs/pids/pretrain_latest.pid"
latest_meta_file="outputs/pids/pretrain_latest.meta"

if [[ "$DEVICE" == "cuda" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

cmd=( "$PYTHON_BIN" -u scripts/run_pretrain.py --config "$CONFIG_PATH" --device "$DEVICE" )

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

echo "Started pretraining"
echo "PID: $pid"
echo "Device: $DEVICE"
if [[ "$DEVICE" == "cuda" ]]; then
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi
echo "Log: $PROJECT_ROOT/$log_file"
echo "Stop: bash scripts/stop_pretrain.sh"
