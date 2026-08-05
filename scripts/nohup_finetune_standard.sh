#!/usr/bin/env bash
# Run HyperFounder finetune on standard datasets (cora, citeseer, pubmed)
# for fair comparison with HGNN baseline

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="${1:-configs/finetune_node_standard.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p outputs/pids outputs/logs

timestamp="$(date +%Y%m%d_%H%M%S)"
config_name="$(basename "$CONFIG_PATH" .yaml)"
log_file="outputs/logs/finetune_standard_${timestamp}.log"
pid_file="outputs/pids/finetune_standard.pid"

cmd=( "$PYTHON_BIN" scripts/run_transfer.py --config "$CONFIG_PATH" --heldout_domain citation )

nohup "${cmd[@]}" > "$log_file" 2>&1 &
pid=$!

sleep 1

echo "$pid" > "$pid_file"

echo "=============================================="
echo "Started finetune on standard datasets"
echo "=============================================="
echo "PID: $pid"
echo "Log: $log_file"
echo "Config: $CONFIG_PATH"
echo ""
echo "To check progress:"
echo "  tail -f $log_file"
echo "  cat outputs/pids/finetune_standard.pid | xargs kill"
