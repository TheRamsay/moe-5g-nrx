#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="${1:-print}"
DATA_ROOT="${DATA_ROOT:-$HOME/moe-5g-datasets/dense-v1}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda}"
WALLTIME="${WALLTIME:-01:00:00}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
HF_DATASET="${HF_DATASET:-Vack0/moe-5g-nrx}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "$CHECKPOINT_DIR" ]]; then
  if [[ "$MODE" == "local" ]]; then
    CHECKPOINT_DIR="checkpoints"
  else
    CHECKPOINT_DIR="../artifacts/checkpoints"
  fi
fi

cd "$REPO_ROOT"

# Safe host-side profiling resources for mixed HF training.
export RESOURCE_PRESET_NAME="gpu-16gb-hf-profile"
export SELECT_RESOURCES="select=1:ncpus=8:ngpus=1:mem=48gb:scratch_ssd=40gb:gpu_mem=16384mb"

echo "HF loader sweep v1"
echo "Study dir: $SCRIPT_DIR"
echo "Mode: $MODE"
echo "HF dataset: $HF_DATASET"
echo "Data root (val): $DATA_ROOT"
echo "Runtime device: $RUNTIME_DEVICE"
echo "Walltime: $WALLTIME"
echo "Select resources: $SELECT_RESOURCES"
echo

build_run_args() {
  local exp_name="$1"
  local workers="$2"
  local prefetch="$3"
  local args=(
    "experiment=exp05_dense_capacity_large"
    "dataset=mixed"
    "runtime.device=$RUNTIME_DEVICE"
    "training.hf_dataset=$HF_DATASET"
    "training.hf_num_workers=$workers"
    "training.hf_prefetch_factor=$prefetch"
    "training.learning_rate=1e-3"
    "training.weight_decay=1e-4"
    "training.max_steps=300"
    "logging.log_every_n_steps=25"
    "validation.enabled=false"
    "training.checkpoint.every_n_steps=0"
    "training.checkpoint.save_latest=false"
    "training.checkpoint.save_best=false"
    "training.checkpoint_dir=$CHECKPOINT_DIR"
    "experiment.batch_name=hf-loader-sweep-v1"
    "experiment.exp_name=$exp_name"
    "experiment.study_slug=2026-04-10-hf-loader-sweep-v1"
    "experiment.study_path=experiments/2026-04-10-hf-loader-sweep-v1"
    "experiment.question=which_hf_loader_worker_prefetch_setting_best_feeds_dense_training"
  )

  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extra_args=( $EXTRA_ARGS )
    args+=("${extra_args[@]}")
  fi

  printf '%s ' "${args[@]}"
}

submit_variant() {
  local exp_name="$1"
  local workers="$2"
  local prefetch="$3"

  case "$MODE" in
    print)
      echo "uv run python main.py $(build_run_args "$exp_name" "$workers" "$prefetch")"
      ;;
    local)
      echo ">>> Running $exp_name locally"
      # shellcheck disable=SC2206
      local run_args=( $(build_run_args "$exp_name" "$workers" "$prefetch") )
      uv run python main.py "${run_args[@]}"
      ;;
    qsub)
      echo ">>> Submitting $exp_name"
      local qsub_args=(-l "walltime=$WALLTIME" -l "$SELECT_RESOURCES")
      qsub "${qsub_args[@]}" -v "RUN_ARGS=$(build_run_args "$exp_name" "$workers" "$prefetch")" scripts/metacentrum_job.sh
      ;;
    *)
      echo "Unknown mode: $MODE"
      echo "Usage: bash submit.sh [print|local|qsub]"
      exit 1
      ;;
  esac
}

submit_variant "hf_loader_w0_p0" 0 0
submit_variant "hf_loader_w1_p1" 1 1
submit_variant "hf_loader_w1_p2" 1 2
submit_variant "hf_loader_w2_p1" 2 1
submit_variant "hf_loader_w2_p2" 2 2
