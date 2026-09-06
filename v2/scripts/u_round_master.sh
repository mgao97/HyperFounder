#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
SLS_PY="/home/user/.conda/envs/sls/bin/python"
mkdir -p outputs_v2/u_checkpoints logs

run_group_seed() {
  local group=$1
  local seed=$2
  shift 2
  local out="$ROOT/outputs_v2/u_checkpoints/${group}/seed_${seed}"
  echo "[u0] group=$group seed=$seed pretrain @ $(date +%F_%T)"
  "$SLS_PY" v2/scripts/run_pretrain_v2.py \
    --config v2/configs/pretrain_v2.yaml \
    --seed "$seed" \
    --output_dir "$out" \
    --checkpoint_fractions "0.12,0.25,0.38,0.50,0.62,0.75,0.88,1.0" \
    "$@" > "logs/u0_train_${group}_seed${seed}.log" 2>&1
}

echo "=== [U0] Generating 8-Interval Checkpoints (Parallel on GPU 3 & 4) ==="
for seed in 1 7 13; do
  # 卡3跑 none 基线, 卡4跑 cca_hor 变体，两组齐头并进
  CUDA_VISIBLE_DEVICES=3 run_group_seed none "$seed" --use_hor false --ablate_cca_card --ablate_cca_film --ablate_cca_tau --ablate_hca_bias --ablate_hca_full &
  CUDA_VISIBLE_DEVICES=4 run_group_seed cca_hor "$seed" --use_hor true --ablate_hca_full &
  wait
done

echo "=== [U1+U2] Running Representation Analysis (GPU 3) ==="
CUDA_VISIBLE_DEVICES=3 "$SLS_PY" v2/scripts/run_u_round.py > logs/u1_u2_analysis.log 2>&1
echo "=== [U-Round] ALL DONE ==="
