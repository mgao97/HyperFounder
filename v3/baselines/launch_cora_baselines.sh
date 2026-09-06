#!/bin/bash
# Launch the 5 Cora baselines across GPUs 4,5,6,7 (GPU 4 hosts 2 models).
# Outputs logs to v3/baselines/logs/ and results to v3/baselines/results/.
set -e

ROOT="/home/user/GSK/mgao/HyperFounder"
PY="/home/user/.conda/envs/sls/bin/python"
BASE="$ROOT/v3/baselines"
mkdir -p "$BASE/logs" "$BASE/results"

# model -> GPU assignment (5 baselines over GPUs 4,5,6,7; GPU 4 runs 2)
declare -A GPU=(
  [mlp]=4
  [hgnn]=4
  [hgnp]=5
  [hnhn]=6
  [unigcn]=7
)

for m in mlp hgnn hgnp hnhn unigcn; do
  GPU_ID=${GPU[$m]}
  echo "Launching cora_$m.py on GPU $GPU_ID ..."
  CUDA_VISIBLE_DEVICES=$GPU_ID \
  OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  nohup "$PY" "$BASE/cora_$m.py" > "$BASE/logs/cora_$m.log" 2>&1 &
  echo "  -> pid $!  (log: $BASE/logs/cora_$m.log)"
done

echo "All 5 Cora baselines launched. Monitor with: tail -f $BASE/logs/cora_*.log"
