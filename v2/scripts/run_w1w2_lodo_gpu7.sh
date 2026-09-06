#!/usr/bin/env bash
# W1-W2 · Leave-1-domain-out linear probe Δ baseline
#
# 环境选择逻辑（自动回退）：
#   1. 用户指定 conda activate sls → 先试 sls 里有没有带 cuda 的 torch
#   2. 若 sls 只有 CPU torch，自动回退到 /home/user/.conda/envs/grag/bin/python
#
# 启动：
#   nohup bash v2/scripts/run_w1w2_lodo_gpu7.sh > logs/w1w2_lodo_gpu7.nohup.log 2>&1 &
#
# 若 pretrain 尚未产出 ckpt，脚本会走 3-step mini-pretrain fallback 把报告链打通；
# ckpt 出现后（outputs_v2/checkpoints/pretrain_best_v2.pt），直接重跑本脚本即可覆盖结果。
set -euo pipefail

ROOT="/home/user/GSK/mgao/HyperFounder"
cd "$ROOT"
mkdir -p logs outputs_v2

# -----------------------------------------------------------------------------
# 1. 按用户要求：conda activate sls
# -----------------------------------------------------------------------------
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sls

export CUDA_VISIBLE_DEVICES=7
export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# -----------------------------------------------------------------------------
# 2. 选 python：sls 里 cuda ok 就用 sls；否则回退到 grag（保证 GPU 真能用上）
# -----------------------------------------------------------------------------
SYS_PY="$(command -v python || true)"
SLS_PY_OK=$("$SYS_PY" -c "import torch; exit(0 if (torch.cuda.is_available() and torch.cuda.device_count()>0) else 1)" 2>/dev/null && echo yes || echo no)
GRAG_PY="/home/user/.conda/envs/grag/bin/python"

if [[ "$SLS_PY_OK" == "yes" ]]; then
  PY_BIN="$SYS_PY"
  DEV_FLAG="cuda:0"
  echo "[W1W2-GPU7] use sls python: $PY_BIN (cuda available)"
else
  echo "[W1W2-GPU7] sls python ($SYS_PY) torch.cuda unavailable — fallback to grag python: $GRAG_PY"
  PY_BIN="$GRAG_PY"
  DEV_FLAG="cuda:0"
fi

# 如果 grag 也在某些环境下识别不出 GPU，就退化为 cpu（保证不挂）
HAS_GPU=$("$PY_BIN" -c "import torch; exit(0 if (torch.cuda.is_available() and torch.cuda.device_count()>0) else 1)" 2>/dev/null && echo yes || echo no)
if [[ "$HAS_GPU" != "yes" ]]; then
  echo "[W1W2-GPU7] WARNING: no GPU visible in current env → fallback to --device cpu"
  DEV_FLAG="cpu"
fi

"$PY_BIN" -c "import torch,sys; print('python:', sys.version.split()[0]); print('torch :', torch.__version__); print('cuda  :', torch.cuda.is_available(), 'devices=', torch.cuda.device_count())"

CKPT="outputs_v2/checkpoints/pretrain_best_v2.pt"
echo "[W1W2-GPU7] START  $(date '+%F %T')  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  device=$DEV_FLAG  ckpt=$CKPT"
"$PY_BIN" v2/scripts/run_w1w2_lodo_linearprobe.py \
    --config v2/configs/pretrain_v2.yaml \
    --pretrain_ckpt "$CKPT" \
    --seeds 1,2,3 \
    --device "$DEV_FLAG" \
    2>&1 | tee -a logs/w1w2_lodo_gpu7.log
EC=${PIPESTATUS[0]}
echo "[W1W2-GPU7] FINISH $(date '+%F %T')  exit=$EC"
exit $EC
