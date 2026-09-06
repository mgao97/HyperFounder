#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/GSK/mgao/HyperFounder"
PY_BIN="${PY_BIN:-python}"
LOG_DIR="$ROOT/logs"
OUT_DIR="${OUT_DIR:-outputs_transferability/mochy_reference_seed7_nohup}"
DATASETS="${DATASETS:-cora_cc,citeseer_cc,coauthorship_cora,coauthorship_dblp,cooking_200,gowalla}"
ENGINE="${ENGINE:-auto}"
PY_THRESH="${PY_THRESH:-6000}"
CPU_THREADS="${CPU_THREADS:-32}"

mkdir -p "$LOG_DIR"

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export PYTHONDONTWRITEBYTECODE=1

LOG_FILE="$LOG_DIR/mochy_reference_$(date +%Y%m%d_%H%M%S).nohup.log"

echo "[mochy-nohup] start @ $(date '+%F %T')"
echo "[mochy-nohup] cpu_threads=$CPU_THREADS engine=$ENGINE out_dir=$OUT_DIR"
echo "[mochy-nohup] log=$LOG_FILE"

nohup "$PY_BIN" "$ROOT/v2/scripts/run_mochy_reference_analysis.py" \
  --datasets "$DATASETS" \
  --engine "$ENGINE" \
  --python_edge_threshold "$PY_THRESH" \
  --output_dir "$OUT_DIR" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$LOG_DIR/mochy_reference_latest.pid"
echo "[mochy-nohup] pid=$PID"
echo "[mochy-nohup] tail -f $LOG_FILE"
