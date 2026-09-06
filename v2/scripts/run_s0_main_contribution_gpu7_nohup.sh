#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

GPU_ID="${GPU_ID:-7}"
SEEDS="${SEEDS:-1,2,3}"
LOG_FILE="$ROOT/logs/s0_main_contribution_master.nohup.log"
PID_FILE="$ROOT/logs/s0_main_contribution_master.pid"

nohup env \
  GPU_ID="$GPU_ID" \
  SEEDS="$SEEDS" \
  bash "$ROOT/v2/scripts/run_s0_main_contribution_check.sh" \
  >"$LOG_FILE" 2>&1 </dev/null &

echo $! > "$PID_FILE"
echo "[launcher] S0 started pid=$(cat "$PID_FILE") gpu=$GPU_ID seeds=$SEEDS"
echo "[launcher] log=$LOG_FILE"
