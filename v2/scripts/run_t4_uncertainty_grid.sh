#!/usr/bin/env bash
# T4:
#   H1 = full + residual uncertainty weighting
#   H2 = CCA+HOR + residual uncertainty weighting

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

mkdir -p logs outputs_v2/t4_uncertainty outputs_v2/figures

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

GPU_ID="${GPU_ID:-7}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
PRETRAIN_SEEDS="${PRETRAIN_SEEDS:-1,7,13}"
EVAL_SEEDS="${EVAL_SEEDS:-1,2,3}"
SLS_PY="/home/user/.conda/envs/sls/bin/python"

run_group_seed() {
  local group="$1"; shift
  local seed="$1"; shift
  local extra=( "$@" )
  local out="$ROOT/outputs_v2/t4_uncertainty/${group}/seed_${seed}"
  local ckpt="$out/checkpoints/pretrain_best_v2.pt"
  mkdir -p "$out"

  echo "[t4] group=$group seed=$seed pretrain @ $(date +%F_%T)"
  "$SLS_PY" v2/scripts/run_pretrain_v2.py \
    --config v2/configs/pretrain_v2.yaml \
    --seed "$seed" \
    --output_dir "$out" \
    --uncertainty_mode residual \
    "${extra[@]}" \
    2>&1 | tee "logs/t4_train_${group}_seed${seed}.log"

  echo "[t4] group=$group seed=$seed frozen probe @ $(date +%F_%T)"
  "$SLS_PY" v2/scripts/run_w1w2_lodo_linearprobe.py \
    --pretrain_ckpt "$ckpt" \
    --seeds "$EVAL_SEEDS" \
    --device cuda:0 \
    --out_csv "outputs_v2/t4_uncertainty/${group}/seed_${seed}/frozen.csv" \
    --variant_tag "${group}_seed${seed}" \
    "${extra[@]}" \
    2>&1 | tee "logs/t4_probe_${group}_seed${seed}.log"
}

IFS=',' read -r -a seed_arr <<< "$PRETRAIN_SEEDS"
for seed in "${seed_arr[@]}"; do
  run_group_seed h1_full_residual_uw "$seed" --use_hor true
  run_group_seed h2_cca_hor_residual_uw "$seed" --ablate_hca_full --use_hor true
done

"$SLS_PY" v2/scripts/summarize_t4_uncertainty.py \
  --in_dir outputs_v2/t4_uncertainty \
  --baseline_dir outputs_v2/t1_ib \
  --out_md logs/t4_uncertainty_summary.md \
  --out_dir outputs_v2/figures

echo "[t4] DONE @ $(date +%F_%T)"
