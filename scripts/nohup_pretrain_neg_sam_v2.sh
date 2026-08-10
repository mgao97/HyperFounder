#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="${1:-configs/pretrain_neg_sam_v2.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
# Default to both 4090s; override via CUDA_VISIBLE_DEVICES if you want a subset.
GPU_ID="${CUDA_VISIBLE_DEVICES:-0,1}"
# Mixed precision: bf16 is recommended for RTX 4090 (Ampere+).
AMP_DTYPE="${AMP_DTYPE:-bf16}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p outputs_neg_sam_v2/logs outputs_neg_sam_v2/pids outputs_neg_sam_v2/checkpoints outputs_neg_sam_v2/results

timestamp="$(date +%Y%m%d_%H%M%S)"
config_name="$(basename "$CONFIG_PATH" .yaml)"
log_file="outputs_neg_sam_v2/logs/${config_name}_${timestamp}.log"
pid_file="outputs_neg_sam_v2/pids/${config_name}_${timestamp}.pid"
latest_pid_file="outputs_neg_sam_v2/pids/pretrain_latest.pid"
latest_meta_file="outputs_neg_sam_v2/pids/pretrain_latest.meta"

if [[ "$DEVICE" == "cuda" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

# Tell torch / NVIDIA to use TF32 + high-precision matmul where possible.
export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

cmd=( "$PYTHON_BIN" -u scripts/run_pretrain_neg_sam.py --config "$CONFIG_PATH" --device "$DEVICE" )

nohup "${cmd[@]}" > "$log_file" 2>&1 &
pid=$!

sleep 2

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
STARTED_AT=$timestamp
AMP_DTYPE=$AMP_DTYPE
EOF

echo "=============================================="
echo "Started pretraining (neg_sam_v2 optimized)"
echo "GPUs: $GPU_ID"
echo "Mixed precision: $AMP_DTYPE"
echo "=============================================="
echo "PID: $pid"
echo "Device: $DEVICE"
echo "CUDA_VISIBLE_DEVICES: $GPU_ID"
echo "Config: $CONFIG_PATH"
echo "Log: $PROJECT_ROOT/$log_file"
echo "Output: outputs_neg_sam_v2/"
echo ""
echo "Key optimizations:"
echo "  - FIXED: loss.backward() + optimizer.step() (was MISSING before)"
echo "  - AMP (bf16) on RTX 4090"
echo "  - Negative-sample caching per (subhg, epoch)"
echo "  - Vectorized overlap sampling + membership BFS"
echo "  - Single CPU sync per subhypergraph-quality eval"
echo "  - Gradient clipping"
echo ""
echo "Stop: bash scripts/stop_pretrain_neg_sam_v2.sh"
echo "Status: bash scripts/status_pretrain_neg_sam_v2.sh"
