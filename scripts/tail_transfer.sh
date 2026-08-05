#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

meta_file="${1:-outputs/pids/transfer_latest.meta}"

if [[ ! -f "$meta_file" ]]; then
  echo "Meta file not found: $meta_file" >&2
  exit 1
fi

log_file=""
while IFS='=' read -r key value; do
  case "$key" in
    LOG) log_file="$value" ;;
  esac
done < "$meta_file"

if [[ -z "$log_file" ]]; then
  echo "LOG entry not found in meta file: $meta_file" >&2
  exit 1
fi

if [[ ! -f "$log_file" ]]; then
  echo "Log file not found: $log_file" >&2
  exit 1
fi

echo "Tracking transfer log: $log_file"
exec tail -n 100 -f "$log_file"
