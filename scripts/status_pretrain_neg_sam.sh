#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

meta_file="${1:-outputs/pids/pretrain_neg_sam_latest.meta}"

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
mode=""
runner=""

while IFS='=' read -r key value; do
  case "$key" in
    PID) pid="$value" ;;
    DEVICE) device="$value" ;;
    CUDA_VISIBLE_DEVICES) cuda_visible_devices="$value" ;;
    CONFIG) config_path="$value" ;;
    LOG) log_file="$value" ;;
    STARTED_AT) started_at="$value" ;;
    MODE) mode="$value" ;;
    RUNNER) runner="$value" ;;
  esac
done < "$meta_file"

echo "Pretrain Neg-Sam Status"
echo "Meta: ${meta_file}"
echo "PID: ${pid:-N/A}"
echo "Device: ${device:-N/A}"
echo "CUDA_VISIBLE_DEVICES: ${cuda_visible_devices:-N/A}"
echo "Config: ${config_path:-N/A}"
echo "Runner: ${runner:-N/A}"
echo "Mode: ${mode:-N/A}"
echo "Log: ${log_file:-N/A}"
echo "Started At: ${started_at:-N/A}"
echo

if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  echo "Process: RUNNING"
  ps -p "$pid" -o pid=,stat=,etime=,pcpu=,pmem=,args=
else
  echo "Process: NOT RUNNING"
fi

if [[ "${device:-}" == "cuda" ]] && command -v nvidia-smi >/dev/null 2>&1 && [[ -n "$pid" ]]; then
  app_line="$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v target_pid="$pid" '$1 == target_pid {print $0; exit}')"
  if [[ -n "$app_line" ]]; then
    gpu_uuid="$(echo "$app_line" | awk -F', *' '{print $2}')"
    gpu_mem="$(echo "$app_line" | awk -F', *' '{print $3}')"
    gpu_line="$(nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader 2>/dev/null | awk -F', *' -v target_uuid="$gpu_uuid" '$3 == target_uuid {print $0; exit}')"
    if [[ -n "$gpu_line" ]]; then
      gpu_index="$(echo "$gpu_line" | awk -F', *' '{print $1}')"
      gpu_name="$(echo "$gpu_line" | awk -F', *' '{print $2}')"
      echo
      echo "GPU:"
      echo "  Index: ${gpu_index}"
      echo "  Name: ${gpu_name}"
      echo "  Memory(MiB): ${gpu_mem}"
    fi
  fi
fi

output_dir="$PROJECT_ROOT/outputs"
if [[ -n "$log_file" ]]; then
  log_dir="$(dirname "$log_file")"
  if [[ "$(basename "$log_dir")" == "logs" ]]; then
    output_dir="$(dirname "$log_dir")"
  fi
fi

step_csv="${output_dir}/logs/pretrain_step_losses_neg_sam.csv"
epoch_csv="${output_dir}/logs/pretrain_losses_neg_sam.csv"
summary_json="${output_dir}/results/pretrain_summary_neg_sam.json"

echo
echo "Recent Loss"
if [[ -f "$step_csv" ]]; then
  echo "Step CSV: $step_csv"
  tail -n 2 "$step_csv"
elif [[ -f "$epoch_csv" ]]; then
  echo "Epoch CSV: $epoch_csv"
  tail -n 2 "$epoch_csv"
else
  echo "No neg-sam loss CSV found under outputs/logs/."
fi

if [[ -f "$summary_json" ]]; then
  echo
  echo "Summary JSON: $summary_json"
fi

if [[ -n "$log_file" && -f "$log_file" ]]; then
  echo
  echo "Recent Log Lines"
  grep -E "Epoch .* step|Epoch .* done|Training finished|New best checkpoint|Early stopping|neg_hyperedges=|neg_memberships=" "$log_file" | tail -n 8 || true
fi
