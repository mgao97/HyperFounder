#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-3}"

SLS_PY="/home/user/.conda/envs/sls/bin/python"

run_group_seed() {
  local group=$1
  local seed=$2
  shift 2
  local extra=("$@")
  local out="$ROOT/outputs_v2/u_checkpoints/${group}/seed_${seed}"
  
  echo "[u0] group=$group seed=$seed pretrain @ $(date +%F_%T)"
  "$SLS_PY" v2/scripts/run_pretrain_v2.py \
    --config v2/configs/pretrain_v2.yaml \
    --seed "$seed" \
    --output_dir "$out" \
    --checkpoint_fractions "0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0" \
    --override_epochs 60 \
    "${extra[@]}" \
    2>&1 | tee "logs/u0_train_${group}_seed${seed}.log"
}

IFS=',' read -r -a seed_arr <<< "${PRETRAIN_SEEDS:-1,7,13}"
for seed in "${seed_arr[@]}"; do
  run_group_seed none "$seed" --use_hor false --ablate_cca_card --ablate_cca_film --ablate_cca_tau --ablate_hca_bias --ablate_hca_full
  run_group_seed cca_hor "$seed" --use_hor true --ablate_hca_full
done

echo "[u0] Checkpoints generation DONE @ $(date +%F_%T)"
