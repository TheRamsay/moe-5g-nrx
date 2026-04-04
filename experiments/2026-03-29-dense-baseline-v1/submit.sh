#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODE="${1:-print}"

EXPERIMENTS=(
  exp01_baseline
)

cd "$REPO_ROOT"

echo "Dense baseline v1"
echo "Study dir: $SCRIPT_DIR"
echo "Mode: $MODE"
echo

case "$MODE" in
  print)
    for exp in "${EXPERIMENTS[@]}"; do
      echo "uv run python main.py experiment=$exp runtime.device=cuda"
    done
    ;;
  local)
    for exp in "${EXPERIMENTS[@]}"; do
      echo ">>> Running $exp locally"
      uv run python main.py experiment="$exp"
    done
    ;;
  qsub)
    for exp in "${EXPERIMENTS[@]}"; do
      echo ">>> Submitting $exp"
      qsub -v "RUN_ARGS=experiment=$exp runtime.device=cuda" scripts/metacentrum_job.sh
    done
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash submit.sh [print|local|qsub]"
    exit 1
    ;;
esac
