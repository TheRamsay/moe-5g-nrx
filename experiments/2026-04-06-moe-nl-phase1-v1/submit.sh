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

echo "MoE NL Phase 1 v1"
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
    "experiment=exp12_moe_nl_phase1_v1"
    "runtime.device=$RUNTIME_DEVICE"
    "training.learning_rate=1e-3"
    "training.weight_decay=1e-4"
    "training.scheduler.name=none"
    "training.max_steps=10000"
    "logging.log_every_n_steps=50"
    "validation.every_n_steps=500"
    "validation.snr_bins=5"
    "validation.data_dir=$DATA_ROOT/val"
    "training.checkpoint_dir=$CHECKPOINT_DIR"
    "experiment.batch_name=moe-nl-phase1-v1"
    "experiment.exp_name=moe_nl_phase1_v1_a1e_3_b0p1_s67"
    "experiment.study_slug=2026-04-06-moe-nl-phase1-v1"
    "experiment.study_path=experiments/2026-04-06-moe-nl-phase1-v1"
    "experiment.question=wider_expert_range_routing_differentiation"
    "model.compute.flops_penalty_alpha=1e-3"
    "model.compute.load_balance_beta=0.1"
  )
  printf '%s ' "${args[@]}"
}

case "$MODE" in
  print)
    echo "uv run python main.py $(build_args)"
    ;;
  local)
    run_args=( $(build_args) )
    uv run python main.py "${run_args[@]}"
    ;;
  qsub)
    echo ">>> Submitting moe_nl_phase1_v1_a1e_3_b0p1_s67"
    qsub_args=(-l "walltime=$WALLTIME")
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
