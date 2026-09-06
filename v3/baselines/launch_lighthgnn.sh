#!/bin/bash
# Launch LightHGNN (HGNN teacher -> MLP student) reproduction on 8 datasets x 5 seeds.
# 4 GPUs, 2 datasets each, seeds run serially per dataset.
set -u
cd /home/user/GSK/mgao/HyperFounder/v3/baselines
LOG=logs/lighthgnn
mkdir -p $LOG

run_gpu() {  # $1=device  $2..=datasets
  local dev=$1; shift
  for ds in "$@"; do
    echo "[$(date +%H:%M:%S)] gpu=$dev ds=$ds start"
    python3 run_lighthgnn.py --teacher hgnn --dataset "$ds" --device "$dev" --num-seeds 5 \
      > "$LOG/lighthgnn_${ds}.log" 2>&1
    echo "[$(date +%H:%M:%S)] gpu=$dev ds=$ds done (exit $?)"
  done
}

run_gpu cuda:4 news20 ca_cora &
run_gpu cuda:5 cc_cora cc_citeseer &
run_gpu cuda:6 dblp4k_paper dblp4k_term &
run_gpu cuda:7 dblp4k_conf imdb_aw &
wait
echo "ALL LightHGNN DONE"
