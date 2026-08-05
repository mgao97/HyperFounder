#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

meta_file="${1:-outputs_neg_sam_v2/pids/pretrain_latest.meta}"

if [[ ! -f "$meta_file" ]]; then
  echo "Meta file not found: $meta_file" >&2
  exit 1
fi

pid=""
device=""
cuda_visible_devices=""
config_path=""
log_file=""
started_at=""

while IFS='=' read -r key value; do
  case "$key" in
    PID) pid="$value" ;;
    DEVICE) device="$value" ;;
    CUDA_VISIBLE_DEVICES) cuda_visible_devices="$value" ;;
    CONFIG) config_path="$value" ;;
    LOG) log_file="$value" ;;
    STARTED_AT) started_at="$value" ;;
  esac
done < "$meta_file"

echo "Pretrain Status (neg_sam_v2)"
echo "=============================================="
echo "Meta: ${meta_file}"
echo "PID: ${pid:-N/A}"
echo "Device: ${device:-N/A}"
echo "CUDA_VISIBLE_DEVICES: ${cuda_visible_devices:-N/A}"
echo "Config: ${config_path:-N/A}"
echo "Log: ${log_file:-N/A}"
echo "Started At: ${started_at:-N/A}"
echo

if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  echo "Process: RUNNING"
  ps -p "$pid" -o pid=,stat=,etime=,pcpu=,pmem=,args=
else
  echo "Process: NOT RUNNING"
fi

step_csv="outputs_neg_sam_v2/logs/pretrain_step_losses.csv"
epoch_csv="outputs_neg_sam_v2/logs/pretrain_losses.csv"
summary_json="outputs_neg_sam_v2/results/pretrain_summary.json"

echo
echo "Recent Loss"
if [[ -f "$step_csv" ]]; then
  echo "Step CSV: $step_csv"
  tail -n 2 "$step_csv"
elif [[ -f "$epoch_csv" ]]; then
  echo "Epoch CSV: $epoch_csv"
  tail -n 2 "$epoch_csv"
else
  echo "No loss CSV found."
fi

if [[ -f "$summary_json" ]]; then
  echo
  echo "Summary JSON: $summary_json"
fi

if [[ -n "$log_file" && -f "$log_file" ]]; then
  echo
  echo "Recent Log Lines"
  grep -E "Epoch .* step|Epoch .* done|Training finished|New best checkpoint|Early stopping|Loss|total=" "$log_file" | tail -n 10 || true
fi
