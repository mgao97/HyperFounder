#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="${1:-}"
HELDOUT_DOMAIN="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -z "$CONFIG_PATH" || -z "$HELDOUT_DOMAIN" ]]; then
  echo "Usage: bash scripts/nohup_transfer.sh <config_path> <heldout_domain>" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p outputs/logs outputs/pids outputs/results

task_name="$(awk -F': *' '/^[[:space:]]*task_name:/ {print $2; exit}' "$CONFIG_PATH" | xargs)"
if [[ -z "$task_name" ]]; then
  echo "task_name not found in config: $CONFIG_PATH" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
config_name="$(basename "$CONFIG_PATH" .yaml)"
heldout_tag="$(echo "$HELDOUT_DOMAIN" | tr '/ ' '__')"
log_file="outputs/logs/${config_name}_${heldout_tag}_transfer_${timestamp}.log"
pid_file="outputs/pids/${config_name}_${heldout_tag}_transfer_${timestamp}.pid"
latest_pid_file="outputs/pids/transfer_latest.pid"
latest_meta_file="outputs/pids/transfer_latest.meta"
result_json="outputs/results/transfer_${task_name}_${HELDOUT_DOMAIN}.json"

if [[ "$DEVICE" == "cuda" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

cmd=( "$PYTHON_BIN" -u scripts/run_transfer.py --config "$CONFIG_PATH" --heldout_domain "$HELDOUT_DOMAIN" )

nohup "${cmd[@]}" > "$log_file" 2>&1 &
pid=$!

sleep 2

if ! ps -p "$pid" > /dev/null 2>&1; then
  echo "Transfer run failed to start. Inspect log: $PROJECT_ROOT/$log_file" >&2
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
TASK=$task_name
HELDOUT_DOMAIN=$HELDOUT_DOMAIN
LOG=$PROJECT_ROOT/$log_file
RESULT_JSON=$PROJECT_ROOT/$result_json
STARTED_AT=$timestamp
RUNNER=$PROJECT_ROOT/scripts/run_transfer.py
EOF

echo "Started transfer run"
echo "PID: $pid"
echo "Task: $task_name"
echo "Heldout Domain: $HELDOUT_DOMAIN"
echo "Device: $DEVICE"
if [[ "$DEVICE" == "cuda" ]]; then
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi
echo "Log: $PROJECT_ROOT/$log_file"
echo "Result: $PROJECT_ROOT/$result_json"
echo "Status: bash scripts/status_transfer.sh"
echo "Tail: bash scripts/tail_transfer.sh"
echo "Stop: bash scripts/stop_transfer.sh"
