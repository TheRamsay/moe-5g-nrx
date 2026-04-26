#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="${1:-print}"
SCENARIO="${SCENARIO:-}"
OUTPUT_DIR="${OUTPUT_DIR:-data}"
DATASET_FOLDER="${DATASET_FOLDER:-./data/deepmimov3/}"
SPLIT="${SPLIT:-test}"
NUM_SAMPLES="${NUM_SAMPLES:-32768}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cpu}"
WALLTIME="${WALLTIME:-00:40:00}"
SELECT_RESOURCES="${SELECT_RESOURCES:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "$SCENARIO" ]]; then
  echo "ERROR: SCENARIO is required."
  echo "Example: SCENARIO=asu_campus1 bash submit.sh print"
  exit 1
fi

cd "$REPO_ROOT"

ENTRYPOINT="scripts/generate_deepmimo_dataset.py"
RUN_ARGS=(
  "generation.split=$SPLIT"
  "generation.num_samples=$NUM_SAMPLES"
  "generation.output_dir=$OUTPUT_DIR"
  "generation.log_to_wandb=true"
  "generation.deepmimo.scenario=$SCENARIO"
  "generation.deepmimo.dataset_folder=$DATASET_FOLDER"
  "runtime.device=$RUNTIME_DEVICE"
)

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args=( $EXTRA_ARGS )
  RUN_ARGS+=("${extra_args[@]}")
fi

build_run_args() {
  printf '%s ' "${RUN_ARGS[@]}"
}

echo "Study: deepmimo-ood-dataset-v1"
echo "Mode: $MODE"
echo "Scenario: $SCENARIO"
echo "Output dir: $OUTPUT_DIR"
echo "DeepMIMO dataset folder: $DATASET_FOLDER"
echo "Split: $SPLIT"
echo "Num samples: $NUM_SAMPLES"
echo

case "$MODE" in
  print)
    echo "uv run python $ENTRYPOINT $(build_run_args)"
    ;;
  local)
    # shellcheck disable=SC2206
    run_args=( $(build_run_args) )
    uv run python "$ENTRYPOINT" "${run_args[@]}"
    ;;
  qsub)
    qsub_args=(-l "walltime=$WALLTIME")
    if [[ -n "$SELECT_RESOURCES" ]]; then
      qsub_args+=(-l "$SELECT_RESOURCES")
    fi
    qsub "${qsub_args[@]}" -v "ENTRYPOINT=$ENTRYPOINT,RUN_ARGS=$(build_run_args)" scripts/metacentrum_job.sh
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash submit.sh [print|local|qsub]"
    exit 1
    ;;
esac
