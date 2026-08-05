#!/usr/bin/env bash
# Run HNN baselines on a single dataset
# Usage: bash run_hnn_single.sh hgnn cora_cc

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${1:-hgnn}"
DATASET="${2:-cora_cc}"

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sls

echo "=============================================="
echo "HNN Baseline Benchmark"
echo "=============================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo ""

# Create results directory
mkdir -p baselines/results

# Run benchmark
python baselines/run_hnn_benchmark.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --num_seeds 3 \
    --hidden_dim 64 \
    --num_layers 2 \
    --dropout 0.5 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --max_epochs 500 \
    --patience 50 \
    --output_dir baselines/results

echo ""
echo "Done! Result saved to baselines/results/${MODEL}_${DATASET}.json"
