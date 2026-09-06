#!/usr/bin/env bash
# ---- Replication run: pretrain seed=3407, 4 key variants (full / no_card / no_HCA / no_HOR) ----
#
# 4 × 60 ep pretrain, each pretrain is followed by LODO probe (5 ds × 3 seeds).
# All variants share the same pretrain seed=3407 for strict cross-variant causal comparison;
# LODO probe downstream seeds 1,2,3 are shared across all variants.
#
# Expected ~ 4 × 10 min (pretrain) + 4 × 15 min (probe) = ~ 100 min total.
#
# Host terminal only. Usage:
#   cd /home/user/GSK/mgao/HyperFounder
#   GPU_ID=0 nohup bash v2/scripts/run_rep_seed3407_4variants.sh \
#       > logs/rep_seed3407_master.nohup.log 2>&1 &
#   echo $! > logs/rep_seed3407_master.pid
#   sleep 5 ; head -60 logs/rep_seed3407_master.nohup.log

set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

mkdir -p logs outputs_v2/rep_seed3407

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1

GPU_ID="${GPU_ID:-7}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

SEED="${PRETRAIN_SEED:-3407}"

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

echo "[rep-seed3407-master] $(date +%F_%T) gpu=$GPU_ID seed=$SEED sls_ok=$SLS_OK py=$PY_BIN"
START_TS="$(date +%s)"

run_variant() {
  local idx="$1"; local total="$2"; local V="$3"; shift 3
  local EXTRA_PRETRAIN=( "$@" )

  local OUT="$ROOT/outputs_v2/rep_seed3407/$V"
  local CKPT="$OUT/checkpoints/pretrain_best_v2.pt"
  local TRAIN_LOG="$ROOT/logs/rep_s3407_train_${V}.log"
  local PROBE_LOG="$ROOT/logs/rep_s3407_probe_${V}.log"
  local PROBE_CSV="$OUT/lodo_probe.csv"
  mkdir -p "$OUT"

  echo ""
  echo "==================================================================="
  echo "[rep-s3407] ($idx/$total) V=$V  pretrain  @ $(date +%F_%T)"
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
  echo "[rep-s3407] ($idx/$total) V=$V pretrain exit=$rc wall=$((t1 - t0))s"

  if [ -f "$CKPT" ]; then
    echo "[rep-s3407] ($idx/$total) V=$V probe  ckpt=$CKPT  @ $(date +%F_%T)"
    local t2 t3
    t2=$(date +%s)
    # shellcheck disable=SC2086
    $probe_script \
        --pretrain_ckpt "$CKPT" \
        --seeds 1,2,3 \
        --device "cuda:0" \
        --out_csv "$PROBE_CSV" \
        --variant_tag "s3407_${V}" \
        "${EXTRA_PRETRAIN[@]}" \
        2>&1 | tee -a "$PROBE_LOG"
    rc=$?
    t3=$(date +%s)
    echo "[rep-s3407] ($idx/$total) V=$V probe exit=$rc wall=$((t3 - t2))s"
  else
    echo "[rep-s3407] ($idx/$total) V=$V SKIP probe (ckpt missing: $CKPT)"
  fi
}

# === 4 KEY VARIANTS ===
run_variant 1 4 full
run_variant 2 4 no_card    --ablate_cca_card
run_variant 3 4 no_HCA     --ablate_hca_full
run_variant 4 4 no_HOR     --use_hor false

END_TS="$(date +%s)"
echo ""
echo "==================================================================="
echo "[rep-s3407-master] ALL 4 DONE. total wall=$((END_TS - START_TS))s"
echo "==================================================================="
for V in full no_card no_HCA no_HOR; do
  sm="$ROOT/outputs_v2/rep_seed3407/$V/lodo_probe_SUMMARY.md"
  best=$(grep -oE "epoch=[0-9]+ loss=[0-9.]+" "$ROOT/logs/rep_s3407_train_${V}.log" 2>/dev/null | tail -1)
  grand=$(grep -oE "Grand mean Δ.*= [+-][0-9.]+" "$sm" 2>/dev/null | head -1)
  echo "  $V  pretrain best=$best  probe $grand"
done
