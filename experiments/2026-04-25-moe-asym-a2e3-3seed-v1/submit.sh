#!/usr/bin/env bash
# Submit 2 multi-seed runs (s32 + s42) on α=2e-3 winner.
# s67 is already done as exp26 in the alpha-sweep study.
# Modes: print | local | qsub

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
  exp28_moe_asym_a2e3_s32
  exp29_moe_asym_a2e3_s42
)

cd "$REPO_ROOT"

echo "Study folder: $SCRIPT_DIR"
echo "Mode: $MODE"
echo "Data root (val/test): $DATA_ROOT"
echo "Train data dir (50k subset): $TRAIN_DATA_DIR"
echo "hf_max_samples: $HF_MAX_SAMPLES"
echo "Walltime: $WALLTIME"
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
      # shellcheck disable=SC2046
      uv run python main.py $(build_run_args "$exp")
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
    exit 1
    ;;
esac
