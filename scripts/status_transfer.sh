#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

meta_file="${1:-outputs/pids/transfer_latest.meta}"

if [[ ! -f "$meta_file" ]]; then
  echo "Meta file not found: $meta_file" >&2
  exit 1
fi

pid=""
device=""
cuda_visible_devices=""
config_path=""
task=""
heldout_domain=""
log_file=""
result_json=""
started_at=""
runner=""

while IFS='=' read -r key value; do
  case "$key" in
    PID) pid="$value" ;;
    DEVICE) device="$value" ;;
    CUDA_VISIBLE_DEVICES) cuda_visible_devices="$value" ;;
    CONFIG) config_path="$value" ;;
    TASK) task="$value" ;;
    HELDOUT_DOMAIN) heldout_domain="$value" ;;
    LOG) log_file="$value" ;;
    RESULT_JSON) result_json="$value" ;;
    STARTED_AT) started_at="$value" ;;
    RUNNER) runner="$value" ;;
  esac
done < "$meta_file"

echo "Transfer Status"
echo "Meta: ${meta_file}"
echo "PID: ${pid:-N/A}"
echo "Task: ${task:-N/A}"
echo "Heldout Domain: ${heldout_domain:-N/A}"
echo "Device: ${device:-N/A}"
echo "CUDA_VISIBLE_DEVICES: ${cuda_visible_devices:-N/A}"
echo "Config: ${config_path:-N/A}"
echo "Runner: ${runner:-N/A}"
echo "Log: ${log_file:-N/A}"
echo "Result JSON: ${result_json:-N/A}"
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

if [[ -n "$result_json" && -f "$result_json" ]]; then
  echo
  echo "Result JSON: $result_json"
fi

if [[ -n "$log_file" && -f "$log_file" ]]; then
  echo
  echo "Recent Log Lines"
  tail -n 20 "$log_file" || true
fi
