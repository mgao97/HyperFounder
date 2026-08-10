#!/usr/bin/env bash
# ============================================================================
# scripts/eval_all.sh — End-to-end evaluation pipeline for neg_sam_v2.
#
# After pretraining has produced outputs_neg_sam_v2/checkpoints/pretrain_best_neg_sam.pt,
# this script runs, in order:
#
#   Step 1. Linear probe: pretrained encoder vs random init, on several datasets.
#   Step 2. Full finetune: pretrained vs scratch, on each held-out domain,
#           across N seeds. Uses configs/finetune_node_v2.yaml and
#           configs/finetune_node_v2_scratch.yaml.
#   Step 3. Baselines: HGNN, HyperGCN, ... (optional, set RUN_BASELINES=1).
#   Step 4. Summary: run scripts/compare_transfer_results.py and write a
#           markdown table under outputs/results/.
#
# Everything is logged to logs/eval_all_<timestamp>.log; each step writes
# its own pid file under pids/. Steps already finished (marker file exists)
# are skipped, so re-running this script resumes safely.
#
# Usage:
#   bash scripts/eval_all.sh                       # use all defaults
#   DEVICE=cuda LINEAR_DATASETS="cora_cc gowalla" bash scripts/eval_all.sh
#   SKIP_LINEAR=1 SKIP_BASELINES=1 bash scripts/eval_all.sh
#
# Env vars:
#   DEVICE                       cuda | cpu                    (default: cuda)
#   PRETRAIN_CKPT                path to pretrained ckpt        (default: outputs_neg_sam_v2/checkpoints/pretrain_best_neg_sam.pt)
#   PRETRAIN_CONFIG              matching pretrain yaml         (default: configs/pretrain_neg_sam_v2.yaml)
#   LINEAR_DATASETS              space-sep names                (default: cora_cc cooking_200 gowalla coauthorship_dblp)
#   LINEAR_SEEDS                 space-sep ints                 (default: 7 13 42)
#   LINEAR_EPOCHS / LINEAR_PATIENCE                              (default: 100 / 20)
#   FINETUNE_HELDOUTS            space-sep domains              (default: citation academic recommendation)
#   FINETUNE_SEEDS               space-sep ints                 (default: 7 13 42)
#   FINETUNE_PRETRAINED_CONFIG   finetune yaml (with ckpt)      (default: configs/finetune_node_v2.yaml)
#   FINETUNE_SCRATCH_CONFIG      finetune yaml (no ckpt)        (default: configs/finetune_node_v2_scratch.yaml)
#   BASELINE_DATASETS            space-sep names                (default: cora citeseer pubmed)
#   BASELINE_MODELS              space-sep names                (default: hgnn hypergcn)
#   RUN_BASELINES                0 | 1                         (default: 0)
#   SKIP_LINEAR / SKIP_FINETUNE / SKIP_BASELINES / SKIP_SUMMARY   (default: 0)
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DEVICE="${DEVICE:-cuda}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-outputs_neg_sam_v2/checkpoints/pretrain_best_neg_sam.pt}"
PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-configs/pretrain_neg_sam_v2.yaml}"
LINEAR_DATASETS="${LINEAR_DATASETS:-cora_cc cooking_200 gowalla coauthorship_dblp}"
LINEAR_SEEDS="${LINEAR_SEEDS:-7 13 42}"
LINEAR_EPOCHS="${LINEAR_EPOCHS:-100}"
LINEAR_PATIENCE="${LINEAR_PATIENCE:-20}"
FINETUNE_HELDOUTS="${FINETUNE_HELDOUTS:-citation academic recommendation}"
FINETUNE_SEEDS="${FINETUNE_SEEDS:-7 13 42}"
FINETUNE_PRETRAINED_CONFIG="${FINETUNE_PRETRAINED_CONFIG:-configs/finetune_node_v2.yaml}"
FINETUNE_SCRATCH_CONFIG="${FINETUNE_SCRATCH_CONFIG:-configs/finetune_node_v2_scratch.yaml}"
BASELINE_DATASETS="${BASELINE_DATASETS:-cora citeseer pubmed}"
BASELINE_MODELS="${BASELINE_MODELS:-hgnn hypergcn}"
RUN_BASELINES="${RUN_BASELINES:-0}"

SKIP_LINEAR="${SKIP_LINEAR:-0}"
SKIP_FINETUNE="${SKIP_FINETUNE:-0}"
SKIP_BASELINES="${SKIP_BASELINES:-0}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"

EVAL_DIR="outputs_neg_sam_v2/eval_all"
mkdir -p "$EVAL_DIR/logs" "$EVAL_DIR/pids" "$EVAL_DIR/markers"

timestamp="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$EVAL_DIR/logs/eval_all_${timestamp}.log"
PID_FILE="$EVAL_DIR/pids/eval_all_${timestamp}.pid"
LATEST_PID_FILE="$EVAL_DIR/pids/eval_all_latest.pid"
LATEST_META="$EVAL_DIR/pids/eval_all_latest.meta"

# Top-level redirect (all stdout/stderr from this point on).
exec > >(tee -a "$LOG_FILE") 2>&1

echo "$BASHPID" > "$PID_FILE"
echo "$BASHPID" > "$LATEST_PID_FILE"

cat > "$LATEST_META" <<EOF
PID=$BASHPID
DEVICE=$DEVICE
PRETRAIN_CKPT=$PRETRAIN_CKPT
STARTED_AT=$timestamp
LOG=$PROJECT_ROOT/$LOG_FILE
EOF

echo "=============================================="
echo "eval_all started"
echo "PID:               $BASHPID"
echo "Device:            $DEVICE"
echo "Pretrained ckpt:   $PRETRAIN_CKPT"
echo "Pretrain config:   $PRETRAIN_CONFIG"
echo "Log file:          $LOG_FILE"
echo "=============================================="

if [[ ! -f "$PRETRAIN_CKPT" ]]; then
  echo "[eval_all] ERROR: pretrained checkpoint not found: $PRETRAIN_CKPT"
  echo "[eval_all] Run pretraining first: bash scripts/nohup_pretrain_neg_sam_v2.sh"
  exit 1
fi

step_marker() { echo "$EVAL_DIR/markers/$1"; }
run_step() {
  local marker_name="$1"; shift
  local marker
  marker="$(step_marker "$marker_name")"
  if [[ -f "$marker" ]]; then
    echo "[eval_all] SKIP step '$marker_name' (marker exists: $marker)"
    return 0
  fi
  echo "[eval_all] START step '$marker_name'"
  if "$@"; then
    echo "[eval_all] DONE  step '$marker_name'"
    touch "$marker"
  else
    echo "[eval_all] FAIL  step '$marker_name' (see $LOG_FILE)"
    return 1
  fi
}

# ============================================================
# Step 1: Linear probe
# ============================================================
if [[ "$SKIP_LINEAR" == "0" ]]; then
  run_step linear_probe bash -c "
    python -u scripts/linear_probe_neg_sam.py \
      --device $DEVICE \
      --pretrained $PRETRAIN_CKPT \
      --config $PRETRAIN_CONFIG \
      --datasets $LINEAR_DATASETS \
      --seeds $LINEAR_SEEDS \
      --epochs $LINEAR_EPOCHS \
      --patience $LINEAR_PATIENCE \
      --output outputs_neg_sam_v2/results/linear_probe_neg_sam.json
  "
else
  echo "[eval_all] SKIP step 'linear_probe' (SKIP_LINEAR=1)"
fi

# ============================================================
# Step 2: Full finetune — pretrained vs scratch, per held-out domain
# ============================================================
if [[ "$SKIP_FINETUNE" == "0" ]]; then
  for domain in $FINETUNE_HELDOUTS; do
    # Pretrained.
    run_step "finetune_pretrained_${domain}" bash -c "
      python -u scripts/run_transfer.py \
        --config $FINETUNE_PRETRAINED_CONFIG \
        --heldout_domain $domain
    "
    # Scratch (same arch, no pretrain).
    run_step "finetune_scratch_${domain}" bash -c "
      python -u scripts/run_transfer.py \
        --config $FINETUNE_SCRATCH_CONFIG \
        --heldout_domain $domain
    "
  done
else
  echo "[eval_all] SKIP step 'finetune' (SKIP_FINETUNE=1)"
fi

# ============================================================
# Step 3: Baselines (HGNN / HyperGCN / ...)
# ============================================================
if [[ "$SKIP_BASELINES" == "0" && "$RUN_BASELINES" == "1" ]]; then
  mkdir -p baselines/results
  for model in $BASELINE_MODELS; do
    for ds in $BASELINE_DATASETS; do
      run_step "baseline_${model}_${ds}" bash -c "
        python -u baselines/run_hnn_benchmark.py \
          --model $model --dataset $ds \
          --num_seeds 3 \
          --output_dir baselines/results
      "
    done
  done
else
  echo "[eval_all] SKIP step 'baselines' (RUN_BASELINES=$RUN_BASELINES, SKIP_BASELINES=$SKIP_BASELINES)"
fi

# ============================================================
# Step 4: Summary
# ============================================================
if [[ "$SKIP_SUMMARY" == "0" ]]; then
  summary_md="outputs_neg_sam_v2/results/v2_vs_scratch_vs_baselines.md"
  run_step summary bash -c "
    python -u scripts/compare_transfer_results.py \
      --results_dir outputs/results \
      --output_markdown $summary_md
    echo '--- summary written to:' $summary_md
  "
else
  echo "[eval_all] SKIP step 'summary' (SKIP_SUMMARY=1)"
fi

echo "=============================================="
echo "eval_all finished. Inspect results:"
echo "  Linear probe:  outputs_neg_sam_v2/results/linear_probe_neg_sam.json"
echo "  Compare table: outputs_neg_sam_v2/results/v2_vs_scratch_vs_baselines.md"
echo "  Full log:      $LOG_FILE"
echo "=============================================="
