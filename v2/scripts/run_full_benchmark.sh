#!/usr/bin/env bash
# Full multi-domain transferability benchmark (motif-safe local set, 8 graphs / 5 domains)
# with 10x degree-preserving null replicates. Runs the python job under nohup so it
# survives even if this wrapper is killed, and prints a heartbeat every 30s to avoid
# idle timeouts.
set -u
cd /home/user/GSK/mgao/HyperFounder

OUT=outputs_transferability/broad_domain_full_seed7
mkdir -p "$OUT"

DATASETS="contact-high-school,contact-primary-school,email-Enron-full,email-Eu-full,coauthorship_cora,coauthorship_dblp,cooking_200,house_committees"
PY=/home/user/.conda/envs/sls/bin/python
LOG="$OUT/run.log"

CUDA_VISIBLE_DEVICES=4 ALLOW_CUDA_PROBE=1 nohup "$PY" v2/scripts/run_broad_domain_benchmark_analysis.py \
    --datasets "$DATASETS" \
    --null_replicates 10 \
    --output_dir "$OUT" \
    --num_threads 16 \
    --skip_missing > "$LOG" 2>&1 &
PID=$!
echo "LAUNCHED PID $PID ; log=$LOG"

while kill -0 "$PID" 2>/dev/null; do
    echo "[HB $(date +%T)] still running (log $(du -h "$LOG" 2>/dev/null | cut -f1))"
    tail -n 3 "$LOG"
    sleep 30
done
wait "$PID"
echo "=== FINISHED exit=$? ==="
tail -n 25 "$LOG"
