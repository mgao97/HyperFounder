#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# scripts/nohup_pretrain_neg_sam_hedg.sh
#
# Background runner for the HEDG-enabled full pretrain.
# Wraps scripts/run_pretrain_neg_sam.py with USE_HEDG_NEGATIVES=1
# and proper log/pid/meta files under outputs_neg_sam_v2/.
#
# Usage:
#   bash scripts/nohup_pretrain_neg_sam_hedg.sh                       # default config
#   bash scripts/nohup_pretrain_neg_sam_hedg.sh configs/pretrain_neg_sam_v2.yaml
#   USE_HEDG=0 bash scripts/nohup_pretrain_neg_sam_hedg.sh           # force 3-mode
#   HEDG_TEMPERATURE=0.3 bash scripts/nohup_pretrain_neg_sam_hedg.sh  # harder negs
#
# Env vars:
#   USE_HEDG              1|0  master switch                  (default: 1)
#   HEDG_TEMPERATURE      float HEDG sampling temperature  (default: 0.5)
#   DEVICE                cpu|cuda                          (default: cuda)
#   GPU_ID                CUDA_VISIBLE_DEVICES              (default: 0,1)
#   AMP_DTYPE             bf16|fp16                         (default: bf16)
# -----------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="${1:-configs/pretrain_neg_sam_v2.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0,1}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
USE_HEDG="${USE_HEDG:-1}"
HEDG_TEMPERATURE="${HEDG_TEMPERATURE:-0.5}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p outputs_neg_sam_v2/logs outputs_neg_sam_v2/pids outputs_neg_sam_v2/checkpoints outputs_neg_sam_v2/results

timestamp="$(date +%Y%m%d_%H%M%S)"
config_name="$(basename "$CONFIG_PATH" .yaml)"
hedg_tag=$([ "$USE_HEDG" = "1" ] && echo "hedg" || echo "3mode")
log_file="outputs_neg_sam_v2/logs/${config_name}_${hedg_tag}_${timestamp}.log"
pid_file="outputs_neg_sam_v2/pids/${config_name}_${hedg_tag}_${timestamp}.pid"
latest_pid_file="outputs_neg_sam_v2/pids/pretrain_${hedg_tag}_latest.pid"
latest_meta_file="outputs_neg_sam_v2/pids/pretrain_${hedg_tag}_latest.meta"

if [[ "$DEVICE" == "cuda" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi
export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export USE_HEDG_NEGATIVES="$USE_HEDG"
# Pass HEDG temperature as a config override: write a temp file patched
# from the original config so the HEDG sampler picks up the temperature.
if [[ "$USE_HEDG" == "1" ]]; then
  patched_config="$PROJECT_ROOT/outputs_neg_sam_v2/logs/${config_name}_${hedg_tag}_${timestamp}.yaml"
  cp "$CONFIG_PATH" "$patched_config"
  python3 -c "
import yaml
with open('$patched_config') as f: cfg = yaml.safe_load(f)
cfg.setdefault('neg_sampling', {}).setdefault('hedg_negatives', {})
cfg['neg_sampling']['hedg_negatives']['enabled'] = True
cfg['neg_sampling']['hedg_negatives']['temperature'] = float('$HEDG_TEMPERATURE')
with open('$patched_config', 'w') as f: yaml.safe_dump(cfg, f)
"
  CONFIG_PATH="$patched_config"
  echo "Patched config written: $patched_config"
  echo "  HEDG temperature: $HEDG_TEMPERATURE"
fi

cmd=( "$PYTHON_BIN" -u scripts/run_pretrain_neg_sam.py --config "$CONFIG_PATH" --device "$DEVICE" )

nohup "${cmd[@]}" > "$log_file" 2>&1 &
pid=$!

sleep 3

if ! ps -p "$pid" > /dev/null 2>&1; then
  echo "Pretraining failed to start. Inspect log: $PROJECT_ROOT/$log_file" >&2
  if [[ -f "$log_file" ]]; then
    tail -n 80 "$log_file" >&2 || true
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
USE_HEDG_NEGATIVES=$USE_HEDG
HEDG_TEMPERATURE=$HEDG_TEMPERATURE
STARTED_AT=$timestamp
EOF

echo "=============================================="
echo "Started pretrain (USE_HEDG_NEGATIVES=$USE_HEDG, tau=$HEDG_TEMPERATURE)"
echo "=============================================="
echo "PID:               $pid"
echo "Device:            $DEVICE"
echo "CUDA_VISIBLE_DEVICES: $GPU_ID"
echo "Config:            $CONFIG_PATH"
echo "Log:               $PROJECT_ROOT/$log_file"
echo "Output:            outputs_neg_sam_v2/"
echo ""
echo "Monitor:"
echo "  tail -f $PROJECT_ROOT/$log_file"
echo "  grep 'Epoch .* done' $PROJECT_ROOT/$log_file | tail -20"
echo "  cat $latest_meta_file"
echo "Stop:"
echo "  kill \$(cat $pid_file)"
