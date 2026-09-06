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
while IFS='=' read -r key value; do
  case "$key" in
    PID) pid="$value" ;;
  esac
done < "$meta_file"

if [[ -z "$pid" ]]; then
  echo "No PID found in meta file" >&2
  exit 1
fi

if kill -0 "$pid" 2>/dev/null; then
  echo "Stopping pretraining process PID=$pid"
  kill "$pid"
  sleep 2
  if kill -0 "$pid" 2>/dev/null; then
    echo "Process still running, sending SIGKILL..."
    kill -9 "$pid"
  fi
  echo "Process stopped."
else
  echo "Process $pid is not running."
fi
