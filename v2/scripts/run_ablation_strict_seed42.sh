#!/usr/bin/env bash
# ---- Route A: strict-seed ablation (pretrain seed=42 for all 8 variants) ----
#
# 8 × 60 epoch pretrain, then 8 × LODO probe.
# Only thing that differs between variants is the model structure switches;
# ALL random sources (init / data shuffle / pretext view sampling / mb sampling)
# are deterministically seeded from the SAME seed=42.
#
# Expected ~ 8 × 10 min = 80 min (pretrain) + ~ 2 h (probe) = 3 h total.
#
# Host terminal only. Usage:
#   cd /home/user/GSK/mgao/HyperFounder
#   nohup bash v2/scripts/run_ablation_strict_seed42.sh \
#       > logs/abl_strict_master.nohup.log 2>&1 &
#   echo $! > logs/abl_strict_master.pid
#   sleep 5 ; head -60 logs/abl_strict_master.nohup.log

set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

mkdir -p logs outputs_v2/ablations_seed42

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1

GPU_ID="${GPU_ID:-7}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

SEED="${PRETRAIN_SEED:-42}"

# ---- Conda env (sls preferred, fallback grag) ----
if [ -f /opt/anaconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /opt/anaconda3/etc/profile.d/conda.sh
fi
if command -v conda >/dev/null 2>&1; then conda activate sls 2>/dev/null || true ; fi

SYS_PY="$(command -v python)"
SLS_OK="no"
if [ -n "$SYS_PY" ]; then
  if "$SYS_PY" -c "import torch; exit(0 if (torch.cuda.is_available() and torch.cuda.device_count()>0) else 1)" 2>/dev/null; then
    SLS_OK="yes"
  fi
fi
GRAG_PY="/home/user/.conda/envs/grag/bin/python"
if [ "$SLS_OK" = "yes" ]; then PY_BIN="$SYS_PY"; else PY_BIN="$GRAG_PY"; fi

probe_script="$PY_BIN v2/scripts/run_w1w2_lodo_linearprobe.py"
pretrain_script="$PY_BIN v2/scripts/run_pretrain_v2.py"

echo "[abl-strict-master] $(date +%F_%T) gpu=$GPU_ID seed=$SEED sls_ok=$SLS_OK py=$PY_BIN"
START_TS="$(date +%s)"

run_variant() {
  local idx="$1"; local total="$2"; local V="$3"; shift 3
  local EXTRA_PRETRAIN=( "$@" )

  local OUT="$ROOT/outputs_v2/ablations_seed42/$V"
  local CKPT="$OUT/checkpoints/pretrain_best_v2.pt"
  local TRAIN_LOG="$ROOT/logs/abl_s42_train_${V}.log"
  local PROBE_LOG="$ROOT/logs/abl_s42_probe_${V}.log"
  local PROBE_CSV="$OUT/lodo_probe.csv"
  mkdir -p "$OUT"

  echo ""
  echo "==================================================================="
  echo "[abl-strict] ($idx/$total) V=$V  pretrain  @ $(date +%F_%T)"
  echo "           out=$OUT"
  echo "           pretrain_extra=(${EXTRA_PRETRAIN[*]:-})"
  echo "==================================================================="
  local t0 t1 rc
  t0=$(date +%s)
  # shellcheck disable=SC2086
  $pretrain_script \
      --config v2/configs/pretrain_v2.yaml \
      --seed "$SEED" \
      --output_dir "$OUT" \
      "${EXTRA_PRETRAIN[@]}" \
      2>&1 | tee -a "$TRAIN_LOG"
  rc=${PIPESTATUS[0]}
  t1=$(date +%s)
  echo "[abl-strict] ($idx/$total) V=$V pretrain exit=$rc wall=$((t1 - t0))s"

  # ---- LODO probe (same fixed seeds 1,2,3 across variants) ----
  if [ -f "$CKPT" ]; then
    echo "[abl-strict] ($idx/$total) V=$V probe  ckpt=$CKPT  @ $(date +%F_%T)"
    local t2 t3
    t2=$(date +%s)
    # shellcheck disable=SC2086
    $probe_script \
        --pretrain_ckpt "$CKPT" \
        --seeds 1,2,3 \
        --device "cuda:0" \
        --out_csv "$PROBE_CSV" \
        --variant_tag "s42_${V}" \
        "${EXTRA_PRETRAIN[@]}" \
        2>&1 | tee -a "$PROBE_LOG"
    rc=$?
    t3=$(date +%s)
    echo "[abl-strict] ($idx/$total) V=$V probe exit=$rc wall=$((t3 - t2))s"
  else
    echo "[abl-strict] ($idx/$total) V=$V SKIP probe (ckpt missing: $CKPT)"
  fi
}

# === W3 CCA rows ===
run_variant 1 8 w3_full
run_variant 2 8 w3_no_card    --ablate_cca_card
run_variant 3 8 w3_no_film    --ablate_cca_film
run_variant 4 8 w3_no_tau     --ablate_cca_tau
# === W4 HCA rows (w4_full = w3_full, skip duplicate) ===
run_variant 5 8 w4_no_bias    --ablate_hca_bias
run_variant 6 8 w4_no_hca     --ablate_hca_full
# === W5 HOR rows ===
run_variant 7 8 w5_with_hor    --use_hor true
run_variant 8 8 w5_without_hor --use_hor false

END_TS="$(date +%s)"
echo ""
echo "==================================================================="
echo "[abl-strict-master] ALL 8 DONE. total wall=$((END_TS - START_TS))s"
echo "==================================================================="
for V in w3_full w3_no_card w3_no_film w3_no_tau w4_no_bias w4_no_hca w5_with_hor w5_without_hor; do
  sm="$ROOT/outputs_v2/ablations_seed42/$V/lodo_probe_SUMMARY.md"
  best=$(grep -oE "epoch=[0-9]+ loss=[0-9.]+" "$ROOT/logs/abl_s42_train_${V}.log" 2>/dev/null | tail -1)
  grand=$(grep -oE "Grand mean Δ.*= [+-][0-9.]+" "$sm" 2>/dev/null | head -1)
  echo "  $V  pretrain best=$best  probe $grand"
done
