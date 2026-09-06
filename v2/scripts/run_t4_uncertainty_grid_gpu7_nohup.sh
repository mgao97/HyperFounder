#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

GPU_ID="${GPU_ID:-7}"
LOG_FILE="$ROOT/logs/t4_uncertainty_master.nohup.log"
PID_FILE="$ROOT/logs/t4_uncertainty_master.pid"

nohup env \
  GPU_ID="$GPU_ID" \
  bash "$ROOT/v2/scripts/run_t4_uncertainty_grid.sh" \
  >"$LOG_FILE" 2>&1 </dev/null &

echo $! > "$PID_FILE"
echo "[launcher] T4 started pid=$(cat "$PID_FILE") gpu=$GPU_ID"
echo "[launcher] log=$LOG_FILE"
