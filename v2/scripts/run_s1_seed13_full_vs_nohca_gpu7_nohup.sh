#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

GPU_ID="${GPU_ID:-7}"
PRETRAIN_SEED="${PRETRAIN_SEED:-13}"
LOG_FILE="$ROOT/logs/s1_seed13_full_vs_nohca_master.nohup.log"
PID_FILE="$ROOT/logs/s1_seed13_full_vs_nohca_master.pid"

nohup env \
  GPU_ID="$GPU_ID" \
  PRETRAIN_SEED="$PRETRAIN_SEED" \
  bash "$ROOT/v2/scripts/run_s1_seed13_full_vs_nohca.sh" \
  >"$LOG_FILE" 2>&1 </dev/null &

echo $! > "$PID_FILE"
echo "[launcher] S1 started pid=$(cat "$PID_FILE") gpu=$GPU_ID pretrain_seed=$PRETRAIN_SEED"
echo "[launcher] log=$LOG_FILE"
