#!/usr/bin/env bash
# v1 re-run: OOM-safe encoder + shared-branch downstream (+ confidence/structural fix in heads_v1)
set -e
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sls

CONFIG=configs/finetune_node_best_v1.yaml
LOG=finetune_citation_v1_run.log

echo "==== v1 finetune start $(date) ====" | tee -a "$LOG"
for DOM in cora citeseer pubmed; do
  echo "==== [v1] dataset=$DOM $(date) ====" | tee -a "$LOG"
  python scripts/run_transfer_v1.py --config "$CONFIG" --heldout_domain "$DOM" 2>&1 | tee -a "$LOG"
done
echo "==== v1 finetune done $(date) ====" | tee -a "$LOG"
