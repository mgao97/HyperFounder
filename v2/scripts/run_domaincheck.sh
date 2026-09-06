#!/usr/bin/env bash
set -u
cd /home/user/GSK/mgao/HyperFounder
OUT=outputs_transferability/broad_domain_domaincheck_seed7
mkdir -p "$OUT"
DATASETS="cooking_200,house_committees,contact-high-school"
PY=/home/user/.conda/envs/sls/bin/python
LOG="$OUT/run.log"
CUDA_VISIBLE_DEVICES=4 ALLOW_CUDA_PROBE=1 nohup "$PY" v2/scripts/run_broad_domain_benchmark_analysis.py --datasets "$DATASETS" --null_replicates 1 --output_dir "$OUT" --num_threads 16 --skip_missing > "$LOG" 2>&1 &
PID=$!
echo "LAUNCHED PID $PID"
while kill -0 "$PID" 2>/dev/null; do echo "[HB $(date +%T)]"; tail -n 2 "$LOG" 2>/dev/null; sleep 30; done
wait "$PID"
echo "=== DONE exit=$? ==="
echo "--- dataset_basic_stats ---"
cat "$OUT/tables/dataset_basic_stats.csv"
