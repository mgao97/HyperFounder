#!/usr/bin/env bash
# W1 · Pretrain GPU 7 (用户要求：优先 GPU 7；严格 spec §3-§5)
#
# 环境选择逻辑（自动回退，保证一定跑起来）：
#   1. 用户指定 conda activate sls → 先试 sls 里有没有带 cuda 的 torch
#   2. 若 sls 只有 CPU torch（常见），自动回退到 /home/user/.conda/envs/grag/bin/python
#      （已验证 1.13.1+cu117 / dhg 缓存/离线 pkl 均可用），继续卡 7。
#
# 启动：
#   nohup bash v2/scripts/run_pretrain_gpu0.sh > logs/pretrain_v2_gpu7.nohup.log 2>&1 &
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

echo "[pretrain-GPU7] CONDA_PREFIX=$CONDA_PREFIX"
echo "[pretrain-GPU7] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# -----------------------------------------------------------------------------
# 2. 选 python：sls 里 cuda ok 就用 sls；否则回退到 grag（保证 GPU 真能用上）
# -----------------------------------------------------------------------------
SYS_PY="$(command -v python || true)"
SLS_PY_OK=$("$SYS_PY" -c "import torch; exit(0 if (torch.cuda.is_available() and torch.cuda.device_count()>0) else 1)" 2>/dev/null && echo yes || echo no)
GRAG_PY="/home/user/.conda/envs/grag/bin/python"

if [[ "$SLS_PY_OK" == "yes" ]]; then
  PY_BIN="$SYS_PY"
  echo "[pretrain-GPU7] use sls python: $PY_BIN (cuda available)"
else
  echo "[pretrain-GPU7] sls python ($SYS_PY) torch.cuda unavailable — fallback to grag python: $GRAG_PY"
  PY_BIN="$GRAG_PY"
fi

"$PY_BIN" -c "import torch,sys; print('python:', sys.version.split()[0]); print('torch :', torch.__version__); print('cuda  :', torch.cuda.is_available(), 'devices=', torch.cuda.device_count())"

echo "[pretrain-GPU7] START  $(date '+%F %T')"
"$PY_BIN" v2/scripts/run_pretrain_v2.py \
    --config v2/configs/pretrain_v2.yaml \
    2>&1 | tee -a logs/pretrain_v2_gpu7.log
EC=${PIPESTATUS[0]}
echo "[pretrain-GPU7] FINISH $(date '+%F %T')  exit=$EC"
exit $EC
