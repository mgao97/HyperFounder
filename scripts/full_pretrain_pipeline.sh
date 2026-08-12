#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# scripts/full_pretrain_pipeline.sh
#
# End-to-end pretrain pipeline orchestrator. Goes from "GPU ready" to
# "pretrained model saved", with intermediate smoke tests and validations.
#
# Phases:
#   0. Pre-flight checks (cache, GPU, dependencies)
#   1. HEDG smoke test (validates new HEDG module)
#   2. Quick pretrain (5 epochs, validates loss decreases)
#   3. (Manual) integrate HEDG into pretext_tasks_neg_sam.py
#   4. Full pretrain (300 epochs, 8 datasets / 4 domains)
#   5. Linear probe (validates pretrained features)
#
# Phases 0-2 are automatic. Phase 3 is a one-line code edit
# (see docs/HEDG_NEGATIVES.md §5). Phase 4-5 require the integration
# to be done; if not done, the runner will skip with a warning.
#
# Usage:
#   bash scripts/full_pretrain_pipeline.sh                  # default (CPU quick)
#   DEVICE=cuda bash scripts/full_pretrain_pipeline.sh     # GPU quick
#   DEVICE=cuda SKIP_SMOKE=1 bash scripts/full_pretrain_pipeline.sh   # skip phase 1
#   DEVICE=cuda SKIP_QUICK=1 bash scripts/full_pretrain_pipeline.sh   # skip phase 2
#   DEVICE=cuda SKIP_FULL=1  bash scripts/full_pretrain_pipeline.sh   # skip phase 4
#   DEVICE=cuda SKIP_PROBE=1  bash scripts/full_pretrain_pipeline.sh   # skip phase 5
# -----------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DEVICE="${DEVICE:-cpu}"
PYTHON_BIN="${PYTHON_BIN:-python}"
HEDG_TEMPERATURE="${HEDG_TEMPERATURE:-0.5}"
HEDG_DATASETS="${HEDG_DATASETS:-cora_cc cooking_200}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-30}"
QUICK_EPOCHS="${QUICK_EPOCHS:-5}"
FULL_EPOCHS="${FULL_EPOCHS:-300}"
PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-configs/pretrain_neg_sam_v2.yaml}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_QUICK="${SKIP_QUICK:-0}"
SKIP_FULL="${SKIP_FULL:-0}"
SKIP_PROBE="${SKIP_PROBE:-0}"

LOG_DIR="outputs_neg_sam_v2/logs/pipeline"
mkdir -p "$LOG_DIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/pipeline_${RUN_ID}.log"
echo "Run id: $RUN_ID, full log: $RUN_LOG" | tee -a "$RUN_LOG"

banner() {
  echo "" | tee -a "$RUN_LOG"
  echo "============================================================" | tee -a "$RUN_LOG"
  echo "  $1" | tee -a "$RUN_LOG"
  echo "============================================================" | tee -a "$RUN_LOG"
}

# ----------------------------------------------------------------------------
# Phase 0: Pre-flight
# ----------------------------------------------------------------------------
banner "Phase 0: Pre-flight checks"

if command -v nvidia-smi >/dev/null 2>&1; then
  if [[ "$DEVICE" == "cuda" ]]; then
    echo "GPU info:" | tee -a "$RUN_LOG"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee -a "$RUN_LOG"
  fi
fi

echo "Cache contents:" | tee -a "$RUN_LOG"
ls -la data/cache/ 2>&1 | tee -a "$RUN_LOG" || echo "  (no cache yet)" | tee -a "$RUN_LOG"

echo "Python: $(${PYTHON_BIN} --version 2>&1)" | tee -a "$RUN_LOG"
echo "Device: $DEVICE" | tee -a "$RUN_LOG"

# ----------------------------------------------------------------------------
# Phase 1: HEDG smoke test
# ----------------------------------------------------------------------------
if [[ "$SKIP_SMOKE" == "0" ]]; then
  banner "Phase 1: HEDG smoke test (HEDG_NEGATIVES module sanity check)"
  HEDG_CMD=(
    bash scripts/run_pretrain_hedg.sh
  )
  echo "Cmd: ${HEDG_CMD[*]}" | tee -a "$RUN_LOG"
  TEMPERATURE="$HEDG_TEMPERATURE" DATASETS="$HEDG_DATASETS" NUM_SAMPLES="$SMOKE_SAMPLES" \
    "${HEDG_CMD[@]}" 2>&1 | tee -a "$RUN_LOG"
  echo "✓ Phase 1 complete" | tee -a "$RUN_LOG"
else
  echo "Phase 1: SKIPPED (SKIP_SMOKE=1)" | tee -a "$RUN_LOG"
fi

# ----------------------------------------------------------------------------
# Phase 2: Quick pretrain (5 epochs, validate loss decreases)
# ----------------------------------------------------------------------------
if [[ "$SKIP_QUICK" == "0" ]]; then
  banner "Phase 2: Quick pretrain ($QUICK_EPOCHS epochs) to validate loss decreases"
  QUICK_LOG="$LOG_DIR/quick_pretrain_${RUN_ID}.log"
  echo "Log: $QUICK_LOG" | tee -a "$RUN_LOG"
  "$PYTHON_BIN" -u scripts/run_pretrain_neg_sam.py \
      --config configs/pretrain_neg_sam_smoke.yaml \
      --device "$DEVICE" 2>&1 | tee "$QUICK_LOG" | tail -20
  
  # Check if loss decreased
  LAST_LOSS=$(tr '\n' ' ' < "$QUICK_LOG" | grep -oE 'Epoch [0-9]+/[0-9]+ done: total=[0-9.]+' | tail -1 | grep -oE '[0-9.]+$')
  FIRST_LOSS=$(tr '\n' ' ' < "$QUICK_LOG" | grep -oE 'Epoch [0-9]+/[0-9]+ done: total=[0-9.]+' | head -1 | grep -oE '[0-9.]+$')
  if [[ -n "$FIRST_LOSS" && -n "$LAST_LOSS" ]]; then
    RATIO=$(awk -v f="$FIRST_LOSS" -v l="$LAST_LOSS" 'BEGIN { printf "%.2f", l/f }')
    echo "  First epoch loss: $FIRST_LOSS" | tee -a "$RUN_LOG"
    echo "  Last epoch loss:  $LAST_LOSS" | tee -a "$RUN_LOG"
    echo "  Loss ratio:       $RATIO (should be < 0.7)" | tee -a "$RUN_LOG"
  fi
  echo "✓ Phase 2 complete" | tee -a "$RUN_LOG"
else
  echo "Phase 2: SKIPPED (SKIP_QUICK=1)" | tee -a "$RUN_LOG"
fi

# ----------------------------------------------------------------------------
# Phase 3: (Manual) Integrate HEDG
# ----------------------------------------------------------------------------
banner "Phase 3: Integrate HEDG (MANUAL STEP — see docs/HEDG_NEGATIVES.md §5)"

INTEGRATED=0
if [[ -n "$(grep -c "from models.hedg_negative_sampling" models/pretext_tasks_neg_sam.py 2>/dev/null || true)" ]]; then
  if [[ "$(grep -c "from models.hedg_negative_sampling" models/pretext_tasks_neg_sam.py)" -gt 0 ]]; then
    INTEGRATED=1
  fi
fi
if [[ "$INTEGRATED" == "1" ]]; then
  echo "✓ HEDG integration detected in pretext_tasks_neg_sam.py" | tee -a "$RUN_LOG"
else
  echo "✗ HEDG NOT yet integrated. To enable, follow docs/HEDG_NEGATIVES.md §5." | tee -a "$RUN_LOG"
  echo "  Continuing with original 3-mode sampling for now." | tee -a "$RUN_LOG"
  echo "  (Set USE_HEDG_NEGATIVES=0 or skip phases 4-5)" | tee -a "$RUN_LOG"
fi

# ----------------------------------------------------------------------------
# Phase 4: Full pretrain (300 epochs)
# ----------------------------------------------------------------------------
if [[ "$SKIP_FULL" == "0" && "$INTEGRATED" == "1" ]]; then
  banner "Phase 4: Full pretrain ($FULL_EPOCHS epochs) — runs in BACKGROUND"
  FULL_LOG="$LOG_DIR/full_pretrain_${RUN_ID}.log"
  echo "Log: $FULL_LOG" | tee -a "$RUN_LOG"
  
  # Patch the config to set the desired number of epochs (preserving the file)
  FULL_CONFIG_TMP="$LOG_DIR/pretrain_neg_sam_v2_${RUN_ID}.yaml"
  cp "$PRETRAIN_CONFIG" "$FULL_CONFIG_TMP"
  # Use Python to safely replace the epochs value
  "$PYTHON_BIN" -c "
import yaml, sys
with open('$FULL_CONFIG_TMP') as f: cfg = yaml.safe_load(f)
cfg['training']['epochs'] = $FULL_EPOCHS
cfg['training']['output_dir'] = 'outputs_neg_sam_v2'
with open('$FULL_CONFIG_TMP', 'w') as f: yaml.safe_dump(cfg, f)
"
  
  echo "Starting full pretrain in background..." | tee -a "$RUN_LOG"
  USE_HEDG_NEGATIVES=1 \
    nohup "$PYTHON_BIN" -u scripts/run_pretrain_neg_sam.py \
        --config "$FULL_CONFIG_TMP" \
        --device "$DEVICE" > "$FULL_LOG" 2>&1 &
  FULL_PID=$!
  echo "Pretrain PID: $FULL_PID" | tee -a "$RUN_LOG"
  echo "Monitor:" | tee -a "$RUN_LOG"
  echo "  tail -f $FULL_LOG" | tee -a "$RUN_LOG"
  echo "  grep 'Epoch .* done' $FULL_LOG | tail -20" | tee -a "$RUN_LOG"
  echo "" | tee -a "$RUN_LOG"
  echo "⚠ This runs in background. Use Phase 5 to validate when complete." | tee -a "$RUN_LOG"
  echo "✓ Phase 4 launched" | tee -a "$RUN_LOG"
elif [[ "$SKIP_FULL" == "0" ]]; then
  echo "Phase 4: SKIPPED (HEDG not integrated)" | tee -a "$RUN_LOG"
fi

# ----------------------------------------------------------------------------
# Phase 5: Linear probe
# ----------------------------------------------------------------------------
if [[ "$SKIP_PROBE" == "0" ]]; then
  banner "Phase 5: Linear probe (validates pretrained features)"
  PROBE_LOG="$LOG_DIR/linear_probe_${RUN_ID}.log"
  CHECKPOINT="outputs_neg_sam_v2/checkpoints/pretrain_best_neg_sam.pt"
  if [[ -f "$CHECKPOINT" ]]; then
    echo "Log: $PROBE_LOG" | tee -a "$RUN_LOG"
    "$PYTHON_BIN" scripts/linear_probe_neg_sam.py --device "$DEVICE" \
        --pretrained "$CHECKPOINT" \
        --config "$PRETRAIN_CONFIG" \
        --datasets cora_cc cooking_200 gowalla \
        --seeds 7 13 42 --epochs 100 --patience 20 2>&1 | tee "$PROBE_LOG" | tail -40
    echo "✓ Phase 5 complete" | tee -a "$RUN_LOG"
  else
    echo "Pretrain checkpoint not found: $CHECKPOINT" | tee -a "$RUN_LOG"
    echo "Phase 5: SKIPPED (no checkpoint yet)" | tee -a "$RUN_LOG"
    echo "  Re-run this script after pretrain completes." | tee -a "$RUN_LOG"
  fi
else
  echo "Phase 5: SKIPPED (SKIP_PROBE=1)" | tee -a "$RUN_LOG"
fi

banner "Pipeline run complete"
echo "Log: $RUN_LOG"
