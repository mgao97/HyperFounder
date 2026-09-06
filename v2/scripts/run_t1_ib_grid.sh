#!/usr/bin/env bash
# T1:
#   G1 = none + IB(beta sweep)
#   G2 = CCA+HOR baseline
#   G3 = CCA+HOR + IB(beta sweep)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

mkdir -p logs outputs_v2/t1_ib outputs_v2/figures

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

GPU_ID="${GPU_ID:-7}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
PRETRAIN_SEEDS="${PRETRAIN_SEEDS:-1,7,13}"
EVAL_SEEDS="${EVAL_SEEDS:-1,2,3}"
BETAS="${BETAS:-1e-4,1e-3,1e-2}"
SLS_PY="/home/user/.conda/envs/sls/bin/python"

run_group_seed() {
  local group="$1"; shift
  local seed="$1"; shift
  local pretrain_extra=( "$@" )
  local probe_extra=()
  local idx=0
  while [ $idx -lt ${#pretrain_extra[@]} ]; do
    local arg="${pretrain_extra[$idx]}"
    case "$arg" in
      --use_ib)
        idx=$((idx + 1))
        continue
        ;;
      --ib_beta|--ib_latent_dim)
        idx=$((idx + 2))
        continue
        ;;
      *)
        probe_extra+=("$arg")
        idx=$((idx + 1))
        ;;
    esac
  done
  local out="$ROOT/outputs_v2/t1_ib/${group}/seed_${seed}"
  local ckpt="$out/checkpoints/pretrain_best_v2.pt"
  mkdir -p "$out"

  echo "[t1] group=$group seed=$seed pretrain @ $(date +%F_%T)"
  "$SLS_PY" v2/scripts/run_pretrain_v2.py \
    --config v2/configs/pretrain_v2.yaml \
    --seed "$seed" \
    --output_dir "$out" \
    "${pretrain_extra[@]}" \
    2>&1 | tee "logs/t1_train_${group}_seed${seed}.log"

  echo "[t1] group=$group seed=$seed frozen probe @ $(date +%F_%T)"
  "$SLS_PY" v2/scripts/run_w1w2_lodo_linearprobe.py \
    --pretrain_ckpt "$ckpt" \
    --seeds "$EVAL_SEEDS" \
    --device cuda:0 \
    --out_csv "outputs_v2/t1_ib/${group}/seed_${seed}/frozen.csv" \
    --variant_tag "${group}_seed${seed}" \
    "${probe_extra[@]}" \
    2>&1 | tee "logs/t1_probe_${group}_seed${seed}.log"
}

IFS=',' read -r -a seed_arr <<< "$PRETRAIN_SEEDS"
IFS=',' read -r -a beta_arr <<< "$BETAS"

for seed in "${seed_arr[@]}"; do
  run_group_seed g2_cca_hor_baseline "$seed" --ablate_hca_full --use_hor true
done

for beta in "${beta_arr[@]}"; do
  for seed in "${seed_arr[@]}"; do
    run_group_seed "g1_none_ib_b${beta}" "$seed" \
      --ablate_cca_card --ablate_cca_film --ablate_cca_tau --ablate_hca_full --use_hor false \
      --use_ib --ib_beta "$beta"
    run_group_seed "g3_cca_hor_ib_b${beta}" "$seed" \
      --ablate_hca_full --use_hor true \
      --use_ib --ib_beta "$beta"
  done
done

"$SLS_PY" v2/scripts/summarize_t1_ib.py \
  --in_dir outputs_v2/t1_ib \
  --none_ref_dir outputs_v2/t5_checkpoint_tradeoff \
  --out_md logs/t1_ib_summary.md \
  --out_dir outputs_v2/figures

echo "[t1] DONE @ $(date +%F_%T)"
