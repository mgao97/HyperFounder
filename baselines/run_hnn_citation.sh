#!/usr/bin/env bash
# Run HNN baselines on citation datasets (cora_cc, citeseer_cc, pubmed_cc)
# This benchmark will compare with HyperFounder's node classification results

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sls

echo "=============================================="
echo "HNN Baseline Benchmark"
echo "=============================================="
echo "Dataset: Citation (cora_cc, citeseer_cc, pubmed_cc)"
echo "Models: hgnn, hnhn, hypergcn, allset, unignn, sheafhypergnn, cegnn"
echo ""

# Create results directory
mkdir -p baselines/results

# Run benchmark for all models on all citation datasets
# Use 3 seeds for statistical significance
python baselines/run_hnn_benchmark.py \
    --model all \
    --dataset citation \
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
echo "Benchmark complete! Results saved to baselines/results/"
echo "Check baselines/results/combined_results.json for all results"
