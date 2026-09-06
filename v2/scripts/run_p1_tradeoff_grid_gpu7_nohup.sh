#!/usr/bin/env bash
# P1-3 Module Tradeoff Grid — nohup starter
#   默认绑定 GPU 7（P0-1 已释放），pretrain_seed=42 (统一种子消融要求)
#   每一行 = pretrain 60ep + frozen probe + finetune
#   预计 8 rows × ~50 min/row ≈ 6.5 h
#
# 关键：使用 bash -lc 启动内层脚本（加载 ~/.bashrc 的 conda 钩子 + sls env），
# 并强制 sls env python 绝对路径，避免 PATH/conda init 失败导致回落到 CPU (grag env)。

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

mkdir -p logs outputs_v2/p1_tradeoff_grid

GPU_ID="${GPU_ID:-7}"
PRETRAIN_SEED="${PRETRAIN_SEED:-42}"
EVAL_SEEDS="${EVAL_SEEDS:-1,2,3}"

LOG_FILE="$ROOT/logs/p1_tradeoff_grid_master.nohup.log"
PID_FILE="$ROOT/logs/p1_tradeoff_grid_master.pid"

# 清理上一次的半截产物（none row 的 cpu 残次 checkpoint）
rm -rf "$ROOT/outputs_v2/p1_tradeoff_grid/none"

nohup bash -lc '
set -euo pipefail
cd "'"$ROOT"'"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
export PYTHONPATH="'"$ROOT"':${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export CUDA_VISIBLE_DEVICES="'"$GPU_ID"'"
export GPU_ID="'"$GPU_ID"'"
export PRETRAIN_SEED="'"$PRETRAIN_SEED"'"
export EVAL_SEEDS="'"$EVAL_SEEDS"'"
# 显式激活 sls 环境，保证 command -v python 指向 sls
source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate sls 2>/dev/null || true
exec bash "'"$ROOT"'/v2/scripts/run_p1_tradeoff_grid.sh"
' >"$LOG_FILE" 2>&1 </dev/null &
echo $! > "$PID_FILE"

sleep 3
PID=$(cat "$PID_FILE")
echo "[p1-grid] started.  master_pid=$PID  gpu=$GPU_ID  pretrain_seed=$PRETRAIN_SEED"
echo "            log = $LOG_FILE"
echo "            out = $ROOT/outputs_v2/p1_tradeoff_grid/"
echo "            rows=8 × [none, CCA, HCA, HOR, CCA_HCA, CCA_HOR, HCA_HOR, full]"
echo ""
echo "  tail -f $LOG_FILE   # to follow progress"

