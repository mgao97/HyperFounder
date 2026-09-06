#!/usr/bin/env bash
# Serial runner for 9 ablation variants (W3×4 + W4×3 + W5×2) × 60 epochs.
# Uses GPU 0 by default; override GPU_ID=7 to run on GPU 7.
#
# Env: conda activate sls first; falls back to /home/user/.conda/envs/grag
# Output: per-variant log at logs/abl_<variant>.log
#         checkpoints at outputs_v2/ablations/<variant>/checkpoints/
#
# Usage (HOST TERMINAL, since sandbox can't write CUDA kernels / proc):
#   cd /home/user/GSK/mgao/HyperFounder
#   nohup bash v2/scripts/run_ablation_9variants.sh > logs/abl_master.nohup.log 2>&1 &
#   echo $! > logs/abl_master.pid
#   sleep 3; tail -30 logs/abl_master.nohup.log

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

mkdir -p logs outputs_v2/ablations

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# ---------- env bootstrap (same as GPU0 pretrain launcher) ----------
if [ -f /opt/anaconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /opt/anaconda3/etc/profile.d/conda.sh
else
  echo "[abl-master] WARNING: /opt/anaconda3/etc/profile.d/conda.sh missing; skipping conda activate"
fi

if command -v conda >/dev/null 2>&1; then
  conda activate sls 2>/dev/null || true
fi

SYS_PY="$(command -v python)"
SLS_OK="no"
if [ -n "$SYS_PY" ]; then
  if "$SYS_PY" -c "import torch; exit(0 if (torch.cuda.is_available() and torch.cuda.device_count()>0) else 1)" 2>/dev/null; then
    SLS_OK="yes"
  fi
fi
GRAG_PY="/home/user/.conda/envs/grag/bin/python"
if [ "$SLS_OK" = "yes" ]; then PY_BIN="$SYS_PY"; else PY_BIN="$GRAG_PY"; fi

export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "[abl-master] $(date +%F_%T) gpu=$GPU_ID sls_ok=$SLS_OK py=$PY_BIN"

# ---------- variant list (9 groups, per design spec §W3–W5) ----------
VARIANTS=(
  # W3 — CCA 4 rows (Cardinality robustness §3 ablation)
  "w3_full"
  "w3_no_card"
  "w3_no_film"
  "w3_no_tau"

  # W4 — HCA 3 rows (Overlap context §4 ablation)
  "w4_no_bias"
  "w4_no_hca"

  # W5 — HOR 2 rows (Higher-order readout §5 ablation)
  "w5_with_hor"
  "w5_without_hor"
)

START_TS="$(date +%s)"
N_VAR="${#VARIANTS[@]}"
IDX=0
for V in "${VARIANTS[@]}"; do
  IDX=$((IDX + 1))
  LOG="logs/abl_${V}.log"
  echo ""
  echo "==================================================================="
  echo "[abl-master] ($IDX/$N_VAR) start variant=$V  log=$LOG  @ $(date +%F_%T)"
  echo "==================================================================="
  t0="$(date +%s)"
  # w3_full 是 w4/w5 的 shared baseline（相同配置），不用再跑一次 w4_full；
  # 如果用户想强制完整，把 VARIANTS 里额外加 "w4_full" 即可。
  "$PY_BIN" v2/scripts/run_ablation.py \
      --variant "$V" \
      --config v2/configs/pretrain_v2.yaml \
      --device "cuda:0" \
      2>&1 | tee -a "$LOG"
  rc=$?
  t1="$(date +%s)"
  echo "[abl-master] ($IDX/$N_VAR) variant=$V  exit=$rc  wall_time_s=$((t1 - t0))  @ $(date +%F_%T)"
done

END_TS="$(date +%s)"
echo ""
echo "==================================================================="
echo "[abl-master] ALL $N_VAR DONE. total_wall_time_s=$((END_TS - START_TS))"
echo "==================================================================="
ls -lat outputs_v2/ablations/ 2>/dev/null
