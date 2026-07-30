#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

target="${1:-outputs/pids/transfer_latest.pid}"

if [[ "$target" =~ ^[0-9]+$ ]]; then
  pid="$target"
else
  if [[ ! -f "$target" ]]; then
    echo "PID file not found: $target" >&2
    exit 1
  fi
  pid="$(cat "$target")"
fi

if ! kill -0 "$pid" 2>/dev/null; then
  echo "Process not running: $pid"
  exit 0
fi

kill "$pid"
echo "Stopped transfer PID: $pid"
