#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# scripts/run_ablation_hedg.sh
#
# Sweep over 6 HEDG ablation configs (3-mode vs 5 HEDG variants) at
# 10 epochs each, save the best loss + per-epoch trajectory.
#
# Usage:
#   bash scripts/run_ablation_hedg.sh
#   EPOCHS=20 bash scripts/run_ablation_hedg.sh
#
# Results written to outputs_ablation_hedg/.
# -----------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

EPOCHS="${EPOCHS:-10}"
DEVICE="${DEVICE:-cpu}"
RESULTS_DIR="outputs_ablation_hedg"
mkdir -p "$RESULTS_DIR/logs" "$RESULTS_DIR/results"

CONFIGS=(
  "ablation_3mode"
  "ablation_hedg_tau05_pert00"
  "ablation_hedg_tau05_pert02"
  "ablation_hedg_tau05_pert05"
  "ablation_hedg_tau01_pert02"
  "ablation_hedg_tau10_pert02"
)

SUMMARY="$RESULTS_DIR/results/summary.csv"
echo "config,best_loss,final_loss,epochs" > "$SUMMARY"

for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "============================================================"
  echo "Running: $cfg  (epochs=$EPOCHS, device=$DEVICE)"
  echo "============================================================"
  OUT_DIR="$RESULTS_DIR/$cfg"
  rm -rf "$OUT_DIR"
  mkdir -p "$OUT_DIR"
  
  # Patch epochs in config
  CONFIG_FILE="configs/ablation_hedg/${cfg}.yaml"
  TMP_CONFIG="$OUT_DIR/${cfg}.yaml"
  cp "$CONFIG_FILE" "$TMP_CONFIG"
  python3 -c "
import yaml
with open('$TMP_CONFIG') as f: cfg = yaml.safe_load(f)
cfg['training']['epochs'] = $EPOCHS
cfg['training']['output_dir'] = '$OUT_DIR'
with open('$TMP_CONFIG', 'w') as f: yaml.safe_dump(cfg, f)
"
  
  LOG="$RESULTS_DIR/logs/${cfg}.log"
  python3 -u scripts/run_pretrain_neg_sam.py \
      --config "$TMP_CONFIG" --device "$DEVICE" > "$LOG" 2>&1
  
  # Extract best + final loss
  BEST=$(grep "Training finished" "$LOG" | grep -oE "best_total=[0-9.]+" | grep -oE "[0-9.]+")
  FINAL=$(grep "Training finished" "$LOG" | grep -oE "train_time_sec=[0-9.]+" | head -1)
  if [[ -z "$BEST" ]]; then
    # Fallback: extract from best checkpoint line
    BEST=$(grep "New best checkpoint" "$LOG" | tail -1 | grep -oE "total=[0-9.]+" | grep -oE "[0-9.]+")
  fi
  if [[ -z "$BEST" ]]; then
    BEST="NaN"
  fi
  # Final epoch loss
  FINAL_LOSS=$(grep "Epoch .* done:" "$LOG" | tail -1 | grep -oE "total=[0-9.]+" | grep -oE "[0-9.]+")
  if [[ -z "$FINAL_LOSS" ]]; then
    FINAL_LOSS="NaN"
  fi
  echo "$cfg,$BEST,$FINAL_LOSS,$EPOCHS" >> "$SUMMARY"
  echo "  → best_loss=$BEST  final_loss=$FINAL_LOSS"
done

echo ""
echo "============================================================"
echo "Ablation sweep complete"
echo "============================================================"
cat "$SUMMARY"
echo ""
echo "Full logs: $RESULTS_DIR/logs/"
echo "Per-config outputs: $RESULTS_DIR/<config>/"
