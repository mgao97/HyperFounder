#!/usr/bin/env bash
# Run HNN baselines with standard parameters from DHG-Bench
# Usage: bash run_standard_benchmark.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sls

echo "=============================================="
echo "HNN Baseline Benchmark (Standard Config)"
echo "=============================================="
echo ""

# Create results directory
mkdir -p baselines/results

# =============================================================================
# HGNN - Standard Config
# =============================================================================
echo ">>> Running HGNN with standard config..."
python baselines/run_hnn_benchmark.py \
    --model hgnn \
    --dataset citation \
    --num_seeds 3 \
    --hidden_dim 128 \
    --num_layers 2 \
    --dropout 0.5 \
    --lr 0.001 \
    --weight_decay 0.0 \
    --max_epochs 100 \
    --patience 50 \
    --output_dir baselines/results

# =============================================================================
# HNHN - Standard Config (Cora)
# =============================================================================
echo ">>> Running HNHN on Cora with standard config..."
python baselines/run_hnn_benchmark.py \
    --model hnhn \
    --dataset cora_cc \
    --num_seeds 3 \
    --hidden_dim 128 \
    --num_layers 1 \
    --dropout 0.5 \
    --lr 0.01 \
    --weight_decay 0.0001 \
    --max_epochs 200 \
    --patience 50 \
    --output_dir baselines/results

# =============================================================================
# HNHN - Standard Config (PubMed)
# =============================================================================
echo ">>> Running HNHN on PubMed with standard config..."
python baselines/run_hnn_benchmark.py \
    --model hnhn \
    --dataset pubmed_cc \
    --num_seeds 3 \
    --hidden_dim 512 \
    --num_layers 1 \
    --dropout 0.5 \
    --lr 0.01 \
    --weight_decay 0.0001 \
    --max_epochs 50 \
    --patience 20 \
    --output_dir baselines/results

echo ""
echo "=============================================="
echo "Benchmark complete!"
echo "=============================================="
echo "Results saved to baselines/results/"
echo ""
echo "Compare with HyperFounder:"
echo "  - HyperFounder (cora_cc): Accuracy = 43.7%"
echo "  - HyperFounder (citeseer_cc): Accuracy = 33.1%"
echo "  - HyperFounder (pubmed_cc): Accuracy = 41.4%"
