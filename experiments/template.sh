#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="${1:-print}"
DATA_ROOT="${DATA_ROOT:-$HOME/moe-5g-datasets/dense-v1}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda}"
WALLTIME="${WALLTIME:-08:00:00}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "$CHECKPOINT_DIR" ]]; then
  if [[ "$MODE" == "local" ]]; then
    CHECKPOINT_DIR="checkpoints"
  else
    CHECKPOINT_DIR="../artifacts/checkpoints"
  fi
fi

# Replace these with the Hydra experiment presets used in this study.
EXPERIMENTS=(
  exp01_baseline
)

cd "$REPO_ROOT"

echo "Study folder: $SCRIPT_DIR"
echo "Mode: $MODE"
echo "Data root: $DATA_ROOT"
echo "Runtime device: $RUNTIME_DEVICE"
echo "Walltime: $WALLTIME"
echo

build_run_args() {
  local exp="$1"
  local args=(
    "experiment=$exp"
    "runtime.device=$RUNTIME_DEVICE"
    "validation.data_dir=$DATA_ROOT/val"
    "training.checkpoint_dir=$CHECKPOINT_DIR"
  )

  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extra_args=( $EXTRA_ARGS )
    args+=("${extra_args[@]}")
  fi

  printf '%s ' "${args[@]}"
}

case "$MODE" in
  print)
    for exp in "${EXPERIMENTS[@]}"; do
      echo "uv run python main.py $(build_run_args "$exp")"
    done
    ;;
  local)
    for exp in "${EXPERIMENTS[@]}"; do
      echo ">>> Running $exp locally"
      # shellcheck disable=SC2206
      run_args=( $(build_run_args "$exp") )
      uv run python main.py "${run_args[@]}"
    done
    ;;
  qsub)
    for exp in "${EXPERIMENTS[@]}"; do
      echo ">>> Submitting $exp"
      qsub -l "walltime=$WALLTIME" -v "RUN_ARGS=$(build_run_args "$exp")" scripts/metacentrum_job.sh
    done
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash submit.sh [print|local|qsub]"
    exit 1
    ;;
esac
