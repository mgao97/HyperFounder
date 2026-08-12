#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# scripts/run_pretrain_hedg.sh
#
# HEDG-Weighted Hard Negative Sampling — runner script.
#
# This script does TWO things, in sequence:
#   1. Runs the HEDG smoke test (scripts/test_hedg_negatives.py) on the
#      smoke datasets. This validates the HEDG sampler builds correctly
#      and the temperature sweep behaves as expected. Takes <1 min on CPU.
#   2. (Optional, controlled by RUN_PRETRAIN env var) Runs a small
#      pretrain using the existing pretrain_neg_sam_v2 pipeline. This
#      is the same pipeline as the smoke pretrain_neg_sam_smoke.yaml,
#      run with configs/pretrain_neg_sam_hedg.yaml.
#
# Neither this script nor the config modify any existing file. The HEDG
# module is loaded fresh from models/hedg_negative_sampling.py.
#
# Env vars:
#   TEMPERATURE       τ for HEDG sampling         (default: 0.5)
#   DATASETS          space-sep dataset names    (default: cora_cc cooking_200)
#   NUM_SAMPLES       pos edges per dataset      (default: 50)
#   RUN_PRETRAIN      0|1, run a small pretrain  (default: 0)
#   PYTHON_BIN        python interpreter         (default: python)
#   DEVICE            cpu|cuda                   (default: cpu)
#   EPOCHS            pretrain epochs            (default: 3)
#
# Usage:
#   bash scripts/run_pretrain_hedg.sh                      # smoke test only
#   TEMPERATURE=0.3 bash scripts/run_pretrain_hedg.sh       # harder negatives
#   RUN_PRETRAIN=1 bash scripts/run_pretrain_hedg.sh        # + small pretrain
#   RUN_PRETRAIN=1 EPOCHS=10 bash scripts/run_pretrain_hedg.sh  # longer pretrain
# -----------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

TEMPERATURE="${TEMPERATURE:-0.5}"
DATASETS="${DATASETS:-cora_cc cooking_200}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
RUN_PRETRAIN="${RUN_PRETRAIN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-3}"

timestamp="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="outputs_neg_sam_hedg/logs"
PID_DIR="outputs_neg_sam_hedg/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

LOG_FILE="$LOG_DIR/hedg_${timestamp}.log"
PID_FILE="$PID_DIR/hedg_${timestamp}.pid"

echo "=============================================="
echo "HEDG-Weighted Hard Negative Sampling"
echo "=============================================="
echo "Temperature:     $TEMPERATURE"
echo "Datasets:        $DATASETS"
echo "Num samples:     $NUM_SAMPLES (smoke)"
echo "Run pretrain:    $RUN_PRETRAIN"
echo "Device:          $DEVICE"
echo "Pretrain epochs: $EPOCHS"
echo "Log file:        $LOG_FILE"
echo "=============================================="
echo ""

# ---------------------------------------------------------------------------
# Step 1: HEDG smoke test (always run)
# ---------------------------------------------------------------------------
echo ">>> Step 1: HEDG smoke test"
SMOKE_CMD=(
  "$PYTHON_BIN" -u scripts/test_hedg_negatives.py
    --datasets $DATASETS
    --temperature "$TEMPERATURE"
    --num-samples "$NUM_SAMPLES"
)
SMOKE_CMD_STR="${SMOKE_CMD[*]}"
echo "    cmd: $SMOKE_CMD_STR"
bash -c "$SMOKE_CMD_STR" 2>&1 | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Step 2: Optional small pretrain
# ---------------------------------------------------------------------------
if [[ "$RUN_PRETRAIN" == "1" ]]; then
  echo ""
  echo ">>> Step 2: Small pretrain with configs/pretrain_neg_sam_hedg.yaml"
  PRETRAIN_LOG="$LOG_DIR/pretrain_${timestamp}.log"
  echo "    log: $PRETRAIN_LOG"
  echo "    Note: this pretrain uses the existing pretrain_neg_sam_v2"
  echo "    pipeline (3-mode sampling) for compatibility. To use the"
  echo "    HEDG sampler inside the training loop, the in-trainer"
  echo "    negative sampler must be swapped (see docs/HEDG_NEGATIVES.md)."
  echo ""

  PRETRAIN_CMD=(
    "$PYTHON_BIN" -u scripts/run_pretrain_neg_sam.py
      --config configs/pretrain_neg_sam_hedg.yaml
      --device "$DEVICE"
  )
  bash -c "${PRETRAIN_CMD[*]}" 2>&1 | tee -a "$PRETRAIN_LOG"
else
  echo ""
  echo ">>> Step 2: skipped (set RUN_PRETRAIN=1 to run a small pretrain)"
fi

echo ""
echo "=============================================="
echo "Done. Log: $LOG_FILE"
echo "=============================================="
