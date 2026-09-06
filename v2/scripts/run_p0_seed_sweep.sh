#!/usr/bin/env bash
# P0-1 seed sweep runner
# - Main table seeds: strict-seed42 already exists in outputs_v2/ablations_seed42/
# - This script fills the remaining planned pretrain seeds {1, 7}
# - Variants: full / no_card / no_HCA / no_HOR + scratch(no-pretrain encoder) control
#
# Host terminal only:
#   cd /home/user/GSK/mgao/HyperFounder
#   GPU_ID=7 nohup bash v2/scripts/run_p0_seed_sweep.sh \
#       > logs/p0_seed_sweep_master.nohup.log 2>&1 &

set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

mkdir -p logs outputs_v2/p0_seed_sweep

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1

GPU_ID="${GPU_ID:-7}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
PRETRAIN_SEEDS="${PRETRAIN_SEEDS:-1 7}"

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

echo "[p0-seed-sweep] $(date +%F_%T) gpu=$GPU_ID seeds=[$PRETRAIN_SEEDS] sls_ok=$SLS_OK py=$PY_BIN"
START_TS="$(date +%s)"

run_variant_for_seed() {
  local seed="$1"; local variant="$2"; shift 2
  local extra=( "$@" )
  local out="$ROOT/outputs_v2/p0_seed_sweep/seed_${seed}/${variant}"
  local ckpt="$out/checkpoints/pretrain_best_v2.pt"
  local tlog="$ROOT/logs/p0_seed_${seed}_train_${variant}.log"
  local plog="$ROOT/logs/p0_seed_${seed}_probe_${variant}.log"
  local pcsv="$out/lodo_probe.csv"
  mkdir -p "$out"

  echo ""
  echo "==================================================================="
  echo "[p0-seed-sweep] seed=$seed  variant=$variant  pretrain @ $(date +%F_%T)"
  echo "                 out=$out"
  echo "                 extra=(${extra[*]:-})"
  echo "==================================================================="
  # shellcheck disable=SC2086
  $pretrain_script \
      --config v2/configs/pretrain_v2.yaml \
      --seed "$seed" \
      --output_dir "$out" \
      "${extra[@]}" \
      2>&1 | tee -a "$tlog"

  echo "[p0-seed-sweep] seed=$seed variant=$variant probe @ $(date +%F_%T)"
  # shellcheck disable=SC2086
  $probe_script \
      --pretrain_ckpt "$ckpt" \
      --seeds 1,2,3 \
      --device "cuda:0" \
      --out_csv "$pcsv" \
      --variant_tag "seed${seed}_${variant}" \
      "${extra[@]}" \
      2>&1 | tee -a "$plog"
}

run_scratch_once() {
  local out="$ROOT/outputs_v2/p0_seed_sweep/scratch"
  local log="$ROOT/logs/p0_scratch_probe.log"
  local csv="$out/lodo_probe.csv"
  mkdir -p "$out"
  echo ""
  echo "==================================================================="
  echo "[p0-seed-sweep] scratch(no-pretrain encoder) probe @ $(date +%F_%T)"
  echo "==================================================================="
  $probe_script \
      --scratch_no_pretrain \
      --pretrain_ckpt "" \
      --seeds 1,2,3 \
      --device "cuda:0" \
      --out_csv "$csv" \
      --variant_tag "scratch_encoder" \
      2>&1 | tee -a "$log"
}

for seed in $PRETRAIN_SEEDS; do
  run_variant_for_seed "$seed" full
  run_variant_for_seed "$seed" no_card --ablate_cca_card
  run_variant_for_seed "$seed" no_HCA  --ablate_hca_full
  run_variant_for_seed "$seed" no_HOR  --use_hor false
done

run_scratch_once

END_TS="$(date +%s)"
echo ""
echo "==================================================================="
echo "[p0-seed-sweep] ALL DONE. total wall=$((END_TS - START_TS))s"
echo "==================================================================="
