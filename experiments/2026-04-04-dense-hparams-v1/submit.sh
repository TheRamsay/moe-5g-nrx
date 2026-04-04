#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="${1:-print}"
BASE_EXPERIMENT="${BASE_EXPERIMENT:-}"
BASE_LABEL="${BASE_LABEL:-}"
DATA_ROOT="${DATA_ROOT:-$HOME/moe-5g-datasets/dense-v1}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda}"
WALLTIME="${WALLTIME:-08:00:00}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
MAX_STEPS="${MAX_STEPS:-10000}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-50}"
VALIDATION_EVERY_N_STEPS="${VALIDATION_EVERY_N_STEPS:-500}"
LEARNING_RATES_STRING="${LEARNING_RATES:-3e-4 1e-3 3e-3}"
WEIGHT_DECAYS_STRING="${WEIGHT_DECAYS:-0 1e-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "$BASE_EXPERIMENT" ]]; then
  echo "BASE_EXPERIMENT is required, e.g. BASE_EXPERIMENT=exp04_dense_capacity_mid"
  exit 1
fi

if [[ -z "$CHECKPOINT_DIR" ]]; then
  if [[ "$MODE" == "local" ]]; then
    CHECKPOINT_DIR="checkpoints"
  else
    CHECKPOINT_DIR="../artifacts/checkpoints"
  fi
fi

case "$BASE_EXPERIMENT" in
  exp01_baseline) DEFAULT_BASE_LABEL="dense_baseline" ;;
  exp03_dense_capacity_small) DEFAULT_BASE_LABEL="dense_small" ;;
  exp04_dense_capacity_mid) DEFAULT_BASE_LABEL="dense_mid" ;;
  exp05_dense_capacity_large) DEFAULT_BASE_LABEL="dense_large" ;;
  *) DEFAULT_BASE_LABEL="$BASE_EXPERIMENT" ;;
esac

if [[ -z "$BASE_LABEL" ]]; then
  BASE_LABEL="$DEFAULT_BASE_LABEL"
fi

IFS=' ' read -r -a LEARNING_RATES <<< "$LEARNING_RATES_STRING"
IFS=' ' read -r -a WEIGHT_DECAYS <<< "$WEIGHT_DECAYS_STRING"

cd "$REPO_ROOT"

echo "Dense hyperparameter sweep v1"
echo "Study dir: $SCRIPT_DIR"
echo "Mode: $MODE"
echo "Base experiment: $BASE_EXPERIMENT"
echo "Base label: $BASE_LABEL"
echo "Data root: $DATA_ROOT"
echo "Runtime device: $RUNTIME_DEVICE"
echo "Walltime: $WALLTIME"
echo "Learning rates: ${LEARNING_RATES[*]}"
echo "Weight decays: ${WEIGHT_DECAYS[*]}"
echo

value_tag() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  value="${value//+/}"
  printf '%s' "$value"
}

build_run_args() {
  local lr="$1"
  local wd="$2"
  local lr_tag wd_tag exp_name
  local args=()

  lr_tag="$(value_tag "$lr")"
  wd_tag="$(value_tag "$wd")"
  exp_name="${BASE_LABEL}_lr${lr_tag}_wd${wd_tag}_s67"

  args=(
    "experiment=$BASE_EXPERIMENT"
    "runtime.device=$RUNTIME_DEVICE"
    "training.learning_rate=$lr"
    "training.weight_decay=$wd"
    "training.max_steps=$MAX_STEPS"
    "logging.log_every_n_steps=$LOG_EVERY_N_STEPS"
    "validation.every_n_steps=$VALIDATION_EVERY_N_STEPS"
    "validation.data_dir=$DATA_ROOT/val"
    "training.checkpoint_dir=$CHECKPOINT_DIR"
    "experiment.batch_name=dense-hparams-v1"
    "experiment.exp_name=$exp_name"
    "experiment.study_slug=2026-04-04-dense-hparams-v1"
    "experiment.study_path=experiments/2026-04-04-dense-hparams-v1"
    "experiment.question=dense_optimizer_tuning_for_selected_capacity"
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
    for lr in "${LEARNING_RATES[@]}"; do
      for wd in "${WEIGHT_DECAYS[@]}"; do
        echo "uv run python main.py $(build_run_args "$lr" "$wd")"
      done
    done
    ;;
  local)
    for lr in "${LEARNING_RATES[@]}"; do
      for wd in "${WEIGHT_DECAYS[@]}"; do
        echo ">>> Running lr=$lr wd=$wd locally"
        # shellcheck disable=SC2206
        run_args=( $(build_run_args "$lr" "$wd") )
        uv run python main.py "${run_args[@]}"
      done
    done
    ;;
  qsub)
    for lr in "${LEARNING_RATES[@]}"; do
      for wd in "${WEIGHT_DECAYS[@]}"; do
        echo ">>> Submitting lr=$lr wd=$wd"
        qsub -l "walltime=$WALLTIME" -v "RUN_ARGS=$(build_run_args "$lr" "$wd")" scripts/metacentrum_job.sh
      done
    done
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash submit.sh [print|local|qsub]"
    exit 1
    ;;
esac
