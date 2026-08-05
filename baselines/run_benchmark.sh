#!/bin/bash
# Run benchmark - using sls environment python

# Set PATH to prioritize sls environment
export PATH="/home/user/.conda/envs/sls/bin:$PATH"

MODEL=${1:-all}
DATASET=${2:-cora}
SEEDS=${3:-3}
OUTPUT_DIR="baselines/results"

echo "========================================"
echo "Running HNN Benchmark"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Seeds: $SEEDS"
echo "Python: $(which python)"
echo "========================================"

cd /home/user/GSK/mgao/HyperFounder

python baselines/run_hnn_benchmark.py \
    --model $MODEL \
    --dataset $DATASET \
    --num_seeds $SEEDS \
    --max_epochs 500 \
    --patience 50 \
    --output_dir $OUTPUT_DIR

echo ""
echo "========================================"
echo "Benchmark completed!"
echo "Results saved to: $OUTPUT_DIR/"
echo "========================================"
