#!/bin/bash
# Launch the 5 baselines (MLP / HGNN / HGNN+ / HNHN / UniGCN) on the 8
# LightHGNN hypergraph datasets (5 models x 8 datasets = 40 runs) across
# GPUs 4,5,6,7. GPU 4 runs mlp (fast) followed by unigcn.
# Logs      -> v3/baselines/logs/hyper/
# Results   -> v3/baselines/results/hyper/<model>_<dataset>.json
set -e

ROOT="/home/user/GSK/mgao/HyperFounder"
PY="/home/user/.conda/envs/sls/bin/python"
BASE="$ROOT/v3/baselines"
LOGS="$BASE/logs/hyper"
mkdir -p "$LOGS" "$BASE/results/hyper"

DATASETS="news20 ca_cora cc_cora cc_citeseer dblp4k_paper dblp4k_term dblp4k_conf imdb_aw"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

run_queue () {  # $1=GPU  $2=model
  local gpu=$1 model=$2
  for ds in $DATASETS; do
    echo "[gpu$gpu] $model / $ds"
    CUDA_VISIBLE_DEVICES=$gpu $PY "$BASE/run_hyper_baseline.py" \
      --model "$model" --dataset "$ds" --device cuda:0 \
      > "$LOGS/${model}_${ds}.log" 2>&1
  done
}

# 5 models over 4 GPUs: GPU4 hosts mlp then unigcn
( run_queue 4 mlp;    run_queue 4 unigcn ) &
run_queue 5 hgnn  &
run_queue 6 hgnnp &
run_queue 7 hnhn  &

wait
echo "All 40 hypergraph baseline runs finished."
echo "Summaries:"
for f in "$BASE/results/hyper"/*.json; do
  python3 -c "
import json,os
d=json.load(open('$f'))
print(f\"  {d['model']:8s} {d['dataset']:14s} val {d['val_acc_mean']*100:.2f}±{d['val_acc_std']*100:.2f}  test {d['test_acc_mean']*100:.2f}±{d['test_acc_std']*100:.2f}\")"
done
