#!/bin/bash
# Finish the remaining hypergraph baseline runs (skips combos whose result
# JSON already exists). Runs one model per GPU on GPUs 4-7:
#   GPU4: hgnn   GPU5: hgnnp   GPU6: hnhn   GPU7: unigcn
# Each model processes the 4 datasets that crashed before the device fix:
# dblp4k_paper, dblp4k_term, dblp4k_conf, imdb_aw.
set -e

ROOT="/home/user/GSK/mgao/HyperFounder"
PY="/home/user/.conda/envs/sls/bin/python"
BASE="$ROOT/v3/baselines"
LOGS="$BASE/logs/hyper"
RES="$BASE/results/hyper"
mkdir -p "$LOGS" "$RES"

DATASETS="dblp4k_paper dblp4k_term dblp4k_conf imdb_aw"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

run_queue () {  # $1=GPU  $2=model
  local gpu=$1 model=$2
  for ds in $DATASETS; do
    if [ -f "$RES/${model}_${ds}.json" ]; then
      echo "[gpu$gpu] $model / $ds -> exists, skip"
      continue
    fi
    echo "[gpu$gpu] $model / $ds"
    CUDA_VISIBLE_DEVICES=$gpu $PY "$BASE/run_hyper_baseline.py" \
      --model "$model" --dataset "$ds" --device cuda:0 \
      > "$LOGS/${model}_${ds}.log" 2>&1
  done
}

run_queue 4 hgnn  &
run_queue 5 hgnnp &
run_queue 6 hnhn  &
run_queue 7 unigcn &

wait
echo "All missing hypergraph baseline runs finished."
