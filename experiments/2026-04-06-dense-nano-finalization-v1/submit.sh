#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="${1:-print}"
DATA_ROOT="${DATA_ROOT:-$HOME/moe-5g-datasets/dense-v1}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda}"
WALLTIME="${WALLTIME:-08:00:00}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
SELECT_RESOURCES="${SELECT_RESOURCES:-}"

if [[ -z "$CHECKPOINT_DIR" ]]; then
  if [[ "$MODE" == "local" ]]; then
    CHECKPOINT_DIR="checkpoints"
  else
    CHECKPOINT_DIR="../artifacts/checkpoints"
  fi
fi

cd "$REPO_ROOT"

echo "Dense nano finalization v1"
echo "Study dir: $SCRIPT_DIR"
echo "Mode: $MODE"
echo "Data root: $DATA_ROOT"
echo "Runtime device: $RUNTIME_DEVICE"
echo "Walltime: $WALLTIME"
if [[ -n "$SELECT_RESOURCES" ]]; then
  echo "Select resources: $SELECT_RESOURCES"
fi
echo

build_args() {
  local args=(
    "experiment=exp11_dense_nano_final20k"
    "runtime.device=$RUNTIME_DEVICE"
    "training.learning_rate=1e-3"
    "training.weight_decay=1e-4"
    "training.scheduler.name=none"
    "training.max_steps=20000"
    "logging.log_every_n_steps=50"
    "validation.every_n_steps=500"
    "validation.snr_bins=5"
    "validation.data_dir=$DATA_ROOT/val"
    "training.checkpoint_dir=$CHECKPOINT_DIR"
    "experiment.batch_name=dense-nano-finalization-v1"
    "experiment.exp_name=dense_nano_final20k_constant_lr_s67"
    "experiment.study_slug=2026-04-06-dense-nano-finalization-v1"
    "experiment.study_path=experiments/2026-04-06-dense-nano-finalization-v1"
    "experiment.question=finalize_nano_dense_20k_for_moe_warmstart"
  )
  printf '%s ' "${args[@]}"
}

case "$MODE" in
  print)
    echo "uv run python main.py $(build_args)"
    ;;
  local)
    echo ">>> Running locally"
    # shellcheck disable=SC2206
    local run_args=( $(build_args) )
    uv run python main.py "${run_args[@]}"
    ;;
  qsub)
    echo ">>> Submitting dense_nano_final20k_constant_lr_s67"
    local qsub_args=(-l "walltime=$WALLTIME")
    if [[ -n "$SELECT_RESOURCES" ]]; then
      qsub_args+=(-l "$SELECT_RESOURCES")
    fi
    qsub "${qsub_args[@]}" -v "RUN_ARGS=$(build_args)" scripts/metacentrum_job.sh
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash submit.sh [print|local|qsub]"
    exit 1
    ;;
esac
