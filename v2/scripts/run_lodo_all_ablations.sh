#!/usr/bin/env bash
# Serial runner: LODO probe 8 ablation variants × 5 ds × 3 seeds.
# Time estimate ~ 6-10 min per variant × 8 variants = 50-80 minutes.
#
# Host terminal only (sandbox can't write CUDA kernels).
# Usage:
#   cd /home/user/GSK/mgao/HyperFounder
#   nohup bash v2/scripts/run_lodo_all_ablations.sh > logs/lodo_abl_master.nohup.log 2>&1 &
#   echo $! > logs/lodo_abl_master.pid
#   sleep 4; tail -30 logs/lodo_abl_master.nohup.log

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

mkdir -p logs outputs_v2/ablations

GPU_ID="${GPU_ID:-7}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

if [ -f /opt/anaconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /opt/anaconda3/etc/profile.d/conda.sh
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

echo "[lodo-abl-master] $(date +%F_%T) gpu=$GPU_ID sls_ok=$SLS_OK py=$PY_BIN"

probe_one() {
  local idx="$1"; local total="$2"; local V="$3"; local CKPT_REL="$4"; shift 4
  local EXTRA=( "$@" )
  local ABS_CKPT="$ROOT/$CKPT_REL"
  local OUT_DIR="$ROOT/outputs_v2/ablations/$V"
  local OUT_CSV="$OUT_DIR/lodo_probe.csv"
  local LOG="$ROOT/logs/lodo_abl_${V}.log"
  mkdir -p "$OUT_DIR"
  echo ""
  echo "==================================================================="
  echo "[lodo-abl-master] ($idx/$total) probe variant=$V  log=$LOG  @ $(date +%F_%T)"
  echo "                   ckpt=$ABS_CKPT"
  echo "                   extra_flags=${EXTRA[*]:-<none>}"
  echo "==================================================================="
  local t0 t1 rc
  t0=$(date +%s)
  if [ ${#EXTRA[@]} -gt 0 ]; then
    "$PY_BIN" v2/scripts/run_w1w2_lodo_linearprobe.py \
        --pretrain_ckpt "$ABS_CKPT" \
        --seeds 1,2,3 \
        --device "cuda:0" \
        --out_csv "$OUT_CSV" \
        --variant_tag "$V" \
        "${EXTRA[@]}" \
        2>&1 | tee -a "$LOG"
    rc=$?
  else
    "$PY_BIN" v2/scripts/run_w1w2_lodo_linearprobe.py \
        --pretrain_ckpt "$ABS_CKPT" \
        --seeds 1,2,3 \
        --device "cuda:0" \
        --out_csv "$OUT_CSV" \
        --variant_tag "$V" \
        2>&1 | tee -a "$LOG"
    rc=$?
  fi
  t1=$(date +%s)
  echo "[lodo-abl-master] ($idx/$total) variant=$V  exit=$rc  wall_time_s=$((t1 - t0))  @ $(date +%F_%T)"
}

START_TS="$(date +%s)"
N=8
# W3 — CCA × 4 rows
probe_one 1  $N w3_full        "outputs_v2/ablations/w3_full/checkpoints/pretrain_best_v2.pt"
probe_one 2  $N w3_no_card     "outputs_v2/ablations/w3_no_card/checkpoints/pretrain_best_v2.pt"      --ablate_cca_card
probe_one 3  $N w3_no_film     "outputs_v2/ablations/w3_no_film/checkpoints/pretrain_best_v2.pt"      --ablate_cca_film
probe_one 4  $N w3_no_tau      "outputs_v2/ablations/w3_no_tau/checkpoints/pretrain_best_v2.pt"       --ablate_cca_tau
# W4 — HCA × 3 rows (w4_full 复用 w3_full，所以 probe w3_full 就够了；这里为了对齐 W4 表头只跑剩下两个)
probe_one 5  $N w4_no_bias     "outputs_v2/ablations/w4_no_bias/checkpoints/pretrain_best_v2.pt"      --ablate_hca_bias
probe_one 6  $N w4_no_hca      "outputs_v2/ablations/w4_no_hca/checkpoints/pretrain_best_v2.pt"       --ablate_hca_full
# W5 — HOR × 2 rows
probe_one 7  $N w5_with_hor    "outputs_v2/ablations/w5_with_hor/checkpoints/pretrain_best_v2.pt"     --use_hor true
probe_one 8  $N w5_without_hor "outputs_v2/ablations/w5_without_hor/checkpoints/pretrain_best_v2.pt"  --use_hor false

END_TS="$(date +%s)"
echo ""
echo "==================================================================="
echo "[lodo-abl-master] ALL $N DONE. total_wall_time_s=$((END_TS - START_TS))"
echo "==================================================================="
# Print final Δ summary from each variant SUMMARY
for V in w3_full w3_no_card w3_no_film w3_no_tau w4_no_bias w4_no_hca w5_with_hor w5_without_hor; do
  sm="$ROOT/outputs_v2/ablations/$V/lodo_probe_SUMMARY.md"
  echo "---- $V ----"
  [ -f "$sm" ] && tail -3 "$sm" || echo "(summary missing)"
done
