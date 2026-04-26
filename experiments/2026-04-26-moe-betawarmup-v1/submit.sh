#!/usr/bin/env bash
# Submit 3-seed β-warmup stabilization study.
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

if [[ -z "$CHECKPOINT_DIR" ]]; then
  if [[ "$MODE" == "local" ]]; then CHECKPOINT_DIR="checkpoints"; else CHECKPOINT_DIR="../artifacts/checkpoints"; fi
fi

EXPERIMENTS=(
  exp35_moe_betawarmup_a2e3_s67
  exp36_moe_betawarmup_a2e3_s32
  exp37_moe_betawarmup_a2e3_s42
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
  local)  for exp in "${EXPERIMENTS[@]}"; do uv run python main.py $(build_run_args "$exp"); done ;;
  qsub)
    for exp in "${EXPERIMENTS[@]}"; do
      echo ">>> Submitting $exp"
      qsub -l "walltime=$WALLTIME" -l "$SELECT_RESOURCES" \
        -v "RUN_ARGS=$(build_run_args "$exp")" scripts/metacentrum_job.sh
    done ;;
  *) echo "Unknown mode: $MODE"; exit 1 ;;
esac
