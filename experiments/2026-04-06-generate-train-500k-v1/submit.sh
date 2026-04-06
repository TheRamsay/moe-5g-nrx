#!/usr/bin/env bash
# Generate 500k-sample training cache for uma and tdlc.
# Stored under ~/moe-5g-datasets/train-500k-v1/{uma,tdlc}.pt
# Purpose: replace on-the-fly Sionna generation during training to improve GPU utilization.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="${1:-print}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cpu}"   # Generation runs fine on CPU
WALLTIME="${WALLTIME:-08:00:00}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/moe-5g-datasets/train-500k-v1}"

cd "$REPO_ROOT"

source experiments/resources/gpu-16gb.sh
# Generation only — override to not waste a 46GB GPU slot
SELECT_RESOURCES="select=1:ncpus=4:ngpus=1:mem=32gb:scratch_ssd=10gb:gpu_mem=16384mb"

echo "Study: generate-train-500k-v1"
echo "Mode: $MODE"
echo "Output dir: $OUTPUT_DIR"
echo "Device: $RUNTIME_DEVICE"
echo

ENTRYPOINT="scripts/generate_datasets.py"
RUN_ARGS="generation.num_samples=500000 generation.batch_size=64 generation.split=train generation.profiles=[uma,tdlc] generation.include_mixed=false generation.output_dir=$OUTPUT_DIR generation.log_to_wandb=false runtime.device=$RUNTIME_DEVICE"

case "$MODE" in
  print)
    echo "uv run python $ENTRYPOINT $RUN_ARGS"
    ;;
  local)
    uv run python $ENTRYPOINT $RUN_ARGS
    ;;
  qsub)
    echo ">>> Submitting generate-train-500k-v1"
    qsub \
      -l "walltime=$WALLTIME" \
      -l "$SELECT_RESOURCES" \
      -v "ENTRYPOINT=$ENTRYPOINT,RUN_ARGS=$RUN_ARGS" \
      scripts/metacentrum_job.sh
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash submit.sh [print|local|qsub]"
    exit 1
    ;;
esac
