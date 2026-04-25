#!/usr/bin/env bash
# Submit the 4-point alpha sweep: exp24..exp27.
# Modes: print (default) | local | qsub
#
# Per CLAUDE.md: always pass validation.data_dir, training.hf_train_data_dir,
# training.hf_max_samples; no gpu_mem in qsub.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="${1:-print}"
DATA_ROOT="${DATA_ROOT:-$HOME/moe-5g-datasets/dense-v1}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$HOME/moe-5g-datasets/train-50k-array3d}"
HF_MAX_SAMPLES="${HF_MAX_SAMPLES:-50000}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda}"
WALLTIME="${WALLTIME:-02:00:00}"
SELECT_RESOURCES="${SELECT_RESOURCES:-select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "$CHECKPOINT_DIR" ]]; then
  if [[ "$MODE" == "local" ]]; then
    CHECKPOINT_DIR="checkpoints"
  else
    CHECKPOINT_DIR="../artifacts/checkpoints"
  fi
fi

EXPERIMENTS=(
  exp24_moe_alphasweep_a5e4
  exp25_moe_alphasweep_a1e3
  exp26_moe_alphasweep_a2e3
  exp27_moe_alphasweep_a5e3
)

cd "$REPO_ROOT"

echo "Study folder: $SCRIPT_DIR"
echo "Mode: $MODE"
echo "Data root (val/test): $DATA_ROOT"
echo "Train data dir (50k subset): $TRAIN_DATA_DIR"
echo "hf_max_samples: $HF_MAX_SAMPLES"
echo "Runtime device: $RUNTIME_DEVICE"
echo "Walltime: $WALLTIME"
echo "Select resources: $SELECT_RESOURCES"
echo

build_run_args() {
  local exp="$1"
  local args=(
    "experiment=$exp"
    "runtime.device=$RUNTIME_DEVICE"
    "validation.data_dir=$DATA_ROOT/val"
    "training.hf_train_data_dir=$TRAIN_DATA_DIR"
    "training.hf_max_samples=$HF_MAX_SAMPLES"
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
      qsub -l "walltime=$WALLTIME" -l "$SELECT_RESOURCES" \
        -v "RUN_ARGS=$(build_run_args "$exp")" scripts/metacentrum_job.sh
    done
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash submit.sh [print|local|qsub]"
    exit 1
    ;;
esac
