#!/usr/bin/env bash
# Run HNN baselines with correct dataset splits
# Using 'cora' (standard splits) instead of 'cora_cc' (cocitation splits)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sls

echo "=============================================="
echo "HNN Baseline Benchmark (Correct Splits)"
echo "=============================================="
echo ""
echo "Dataset splits comparison:"
echo "  - cora_cc: Train=5.17%, Val=94.83%, Test=94.83% (WRONG - val/test overlap)"
echo "  - cora: Standard 48/32/20 splits (CORRECT)"
echo ""

# Create results directory
mkdir -p baselines/results

# =============================================================================
# HGNN - Correct Config for Cora (standard splits)
# =============================================================================
echo ">>> Running HGNN on Cora (standard splits)..."
python baselines/run_hnn_benchmark.py \
    --model hgnn \
    --dataset cora \
    --num_seeds 3 \
    --hidden_dim 128 \
    --num_layers 2 \
    --dropout 0.5 \
    --lr 0.001 \
    --weight_decay 0.0 \
    --max_epochs 100 \
    --patience 50 \
    --output_dir baselines/results

echo ""
echo "=============================================="
echo "Benchmark complete!"
echo "=============================================="
