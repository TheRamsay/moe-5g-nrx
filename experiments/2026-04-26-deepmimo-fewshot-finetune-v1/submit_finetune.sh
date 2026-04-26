#!/usr/bin/env bash
# Stage 2: fine-tune dense_large + exp26 on asu_campus1.
# Requires Stage 1 (data/train/asu_campus1/) to already exist on the cluster
# (generated via the dataset-v1 study with SPLIT=train NUM_SAMPLES=2048).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="${1:-print}"
DATA_ROOT="${DATA_ROOT:-$HOME/moe-5g-datasets/dense-v1}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$REPO_ROOT/data/train}"
HF_MAX_SAMPLES="${HF_MAX_SAMPLES:-2048}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda}"
WALLTIME="${WALLTIME:-01:30:00}"
SELECT_RESOURCES="${SELECT_RESOURCES:-select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=20gb}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-../artifacts/checkpoints}"

EXPERIMENTS=(
  exp50_finetune_dense_large_ood
  exp51_finetune_exp26_ood
)

cd "$REPO_ROOT"

build_run_args() {
  local exp="$1"
  printf '%s ' \
    "experiment=$exp" \
    "runtime.device=$RUNTIME_DEVICE" \
    "validation.data_dir=$DATA_ROOT/val" \
    "training.hf_train_data_dir=$TRAIN_DATA_DIR" \
    "training.hf_max_samples=$HF_MAX_SAMPLES" \
    "training.checkpoint_dir=$CHECKPOINT_DIR"
}

case "$MODE" in
  print)  for exp in "${EXPERIMENTS[@]}"; do echo "uv run python main.py $(build_run_args "$exp")"; done ;;
  qsub)
    for exp in "${EXPERIMENTS[@]}"; do
      echo ">>> $exp"
      qsub -l "walltime=$WALLTIME" -l "$SELECT_RESOURCES" \
        -v "RUN_ARGS=$(build_run_args "$exp")" scripts/metacentrum_job.sh
    done ;;
  *) echo "Unknown mode: $MODE"; exit 1 ;;
esac
