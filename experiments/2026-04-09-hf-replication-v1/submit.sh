#!/usr/bin/env bash
# Replicate dense large baseline using HuggingFace training data (Vack0/moe-5g-nrx)
# to validate dataset quality before using it for Phase 2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="${1:-print}"
DATA_ROOT="${DATA_ROOT:-$HOME/moe-5g-datasets/dense-v1}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda}"
WALLTIME="${WALLTIME:-08:00:00}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
HF_DATASET="${HF_DATASET:-Vack0/moe-5g-nrx}"

if [[ -z "$CHECKPOINT_DIR" ]]; then
  if [[ "$MODE" == "local" ]]; then
    CHECKPOINT_DIR="checkpoints"
  else
    CHECKPOINT_DIR="../artifacts/checkpoints"
  fi
fi

cd "$REPO_ROOT"

# Load gpu-16gb resource preset for dense runs
source experiments/resources/gpu-16gb.sh

echo "HF dataset replication v1"
echo "Study dir: $SCRIPT_DIR"
echo "Mode: $MODE"
echo "HF dataset: $HF_DATASET"
echo "Data root (val): $DATA_ROOT"
echo "Runtime device: $RUNTIME_DEVICE"
echo "Walltime: $WALLTIME"
echo

build_run_args() {
  local exp_name="$1"
  local args=(
    "experiment=exp05_dense_capacity_large"
    "runtime.device=$RUNTIME_DEVICE"
    "training.hf_dataset=$HF_DATASET"
    "training.learning_rate=1e-3"
    "training.weight_decay=1e-4"
    "training.max_steps=10000"
    "logging.log_every_n_steps=50"
    "validation.every_n_steps=500"
    "validation.data_dir=$DATA_ROOT/val"
    "training.checkpoint_dir=$CHECKPOINT_DIR"
    "experiment.batch_name=hf-replication-v1"
    "experiment.exp_name=$exp_name"
    "experiment.study_slug=2026-04-09-hf-replication-v1"
    "experiment.study_path=experiments/2026-04-09-hf-replication-v1"
    "experiment.question=does_hf_training_data_replicate_sionna_baseline"
  )
  printf '%s ' "${args[@]}"
}

submit_variant() {
  local exp_name="$1"

  case "$MODE" in
    print)
      echo "uv run python main.py $(build_run_args "$exp_name")"
      ;;
    local)
      echo ">>> Running $exp_name locally"
      # shellcheck disable=SC2206
      local run_args=( $(build_run_args "$exp_name") )
      uv run python main.py "${run_args[@]}"
      ;;
    qsub)
      echo ">>> Submitting $exp_name"
      local qsub_args=(-l "walltime=$WALLTIME" -l "$SELECT_RESOURCES")
      qsub "${qsub_args[@]}" -v "RUN_ARGS=$(build_run_args "$exp_name")" scripts/metacentrum_job.sh
      ;;
    *)
      echo "Unknown mode: $MODE"
      echo "Usage: bash submit.sh [print|local|qsub]"
      exit 1
      ;;
  esac
}

submit_variant "dense_large_hf_replication_10k_s67"
