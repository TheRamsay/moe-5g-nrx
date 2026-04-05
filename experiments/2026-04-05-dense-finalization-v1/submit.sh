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
COSINE_MIN_LR="${COSINE_MIN_LR:-1e-5}"

if [[ -z "$CHECKPOINT_DIR" ]]; then
  if [[ "$MODE" == "local" ]]; then
    CHECKPOINT_DIR="checkpoints"
  else
    CHECKPOINT_DIR="../artifacts/checkpoints"
  fi
fi

cd "$REPO_ROOT"

echo "Dense finalization v1"
echo "Study dir: $SCRIPT_DIR"
echo "Mode: $MODE"
echo "Data root: $DATA_ROOT"
echo "Runtime device: $RUNTIME_DEVICE"
echo "Walltime: $WALLTIME"
if [[ -n "$SELECT_RESOURCES" ]]; then
  echo "Select resources: $SELECT_RESOURCES"
fi
echo "Cosine min lr: $COSINE_MIN_LR"
echo

build_common_args() {
  local exp_name="$1"
  local scheduler_name="$2"
  local scheduler_extra=()

  if [[ "$scheduler_name" == "cosine" ]]; then
    scheduler_extra=(
      "training.scheduler.name=cosine"
      "training.scheduler.min_lr=$COSINE_MIN_LR"
    )
  else
    scheduler_extra=("training.scheduler.name=none")
  fi

  local args=(
    "experiment=exp05_dense_capacity_large"
    "runtime.device=$RUNTIME_DEVICE"
    "training.learning_rate=1e-3"
    "training.weight_decay=1e-4"
    "training.max_steps=20000"
    "logging.log_every_n_steps=50"
    "validation.every_n_steps=500"
    "validation.data_dir=$DATA_ROOT/val"
    "training.checkpoint_dir=$CHECKPOINT_DIR"
    "experiment.batch_name=dense-finalization-v1"
    "experiment.exp_name=$exp_name"
    "experiment.study_slug=2026-04-05-dense-finalization-v1"
    "experiment.study_path=experiments/2026-04-05-dense-finalization-v1"
    "experiment.question=dense_finalization_longer_training_and_scheduler"
  )
  args+=("${scheduler_extra[@]}")

  printf '%s ' "${args[@]}"
}

submit_variant() {
  local exp_name="$1"
  local scheduler_name="$2"

  case "$MODE" in
    print)
      echo "uv run python main.py $(build_common_args "$exp_name" "$scheduler_name")"
      ;;
    local)
      echo ">>> Running $exp_name locally"
      # shellcheck disable=SC2206
      local run_args=( $(build_common_args "$exp_name" "$scheduler_name") )
      uv run python main.py "${run_args[@]}"
      ;;
    qsub)
      echo ">>> Submitting $exp_name"
      local qsub_args=(-l "walltime=$WALLTIME")
      if [[ -n "$SELECT_RESOURCES" ]]; then
        qsub_args+=(-l "$SELECT_RESOURCES")
      fi
      qsub "${qsub_args[@]}" -v "RUN_ARGS=$(build_common_args "$exp_name" "$scheduler_name")" scripts/metacentrum_job.sh
      ;;
    *)
      echo "Unknown mode: $MODE"
      echo "Usage: bash submit.sh [print|local|qsub]"
      exit 1
      ;;
  esac
}

submit_variant "dense_large_final20k_constant_lr_s67" "none"
submit_variant "dense_large_final20k_cosine_lr_s67" "cosine"
