#!/usr/bin/env bash
# T5: full vs none, pretrain seeds {1,7,13}, checkpoint fractions 25/50/75.
# Outputs:
#   outputs_v2/t5_checkpoint_tradeoff/{full,none}/seed_{s}/...
#   logs/t5_checkpoint_tradeoff_summary.md

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

mkdir -p logs outputs_v2/t5_checkpoint_tradeoff outputs_v2/figures

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

GPU_ID="${GPU_ID:-7}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
PRETRAIN_SEEDS="${PRETRAIN_SEEDS:-1,7,13}"
EVAL_SEEDS="${EVAL_SEEDS:-1,2,3}"
CKPT_FRACS="${CKPT_FRACS:-0.25,0.5,0.75}"
SLS_PY="/home/user/.conda/envs/sls/bin/python"

run_variant_seed() {
  local variant="$1"; shift
  local seed="$1"; shift
  local extra=( "$@" )
  local out="$ROOT/outputs_v2/t5_checkpoint_tradeoff/${variant}/seed_${seed}"
  mkdir -p "$out"

  echo "[t5] variant=$variant seed=$seed pretrain @ $(date +%F_%T)"
  "$SLS_PY" v2/scripts/run_pretrain_v2.py \
    --config v2/configs/pretrain_v2.yaml \
    --seed "$seed" \
    --output_dir "$out" \
    --checkpoint_fractions "$CKPT_FRACS" \
    "${extra[@]}" \
    2>&1 | tee "logs/t5_train_${variant}_seed${seed}.log"

  for tag in 25 50 75; do
    local ckpt="$out/checkpoints/pretrain_frac_${tag}_v2.pt"
    if [ -f "$ckpt" ]; then
      echo "[t5] variant=$variant seed=$seed frac=$tag probe @ $(date +%F_%T)"
      "$SLS_PY" v2/scripts/run_w1w2_lodo_linearprobe.py \
        --pretrain_ckpt "$ckpt" \
        --seeds "$EVAL_SEEDS" \
        --device cuda:0 \
        --out_csv "outputs_v2/t5_checkpoint_tradeoff/${variant}/seed_${seed}/probe_frac_${tag}.csv" \
        --variant_tag "t5_${variant}_seed${seed}_frac${tag}" \
        "${extra[@]}" \
        2>&1 | tee "logs/t5_probe_${variant}_seed${seed}_frac${tag}.log"
    fi
  done
}

IFS=',' read -r -a seed_arr <<< "$PRETRAIN_SEEDS"
for seed in "${seed_arr[@]}"; do
  run_variant_seed full "$seed"
  run_variant_seed none "$seed" --ablate_cca_card --ablate_cca_film --ablate_cca_tau --ablate_hca_full --use_hor false
done

"$SLS_PY" v2/scripts/summarize_t5_checkpoint_tradeoff.py \
  --in_dir outputs_v2/t5_checkpoint_tradeoff \
  --out_md logs/t5_checkpoint_tradeoff_summary.md \
  --out_dir outputs_v2/figures

echo "[t5] DONE @ $(date +%F_%T)"
