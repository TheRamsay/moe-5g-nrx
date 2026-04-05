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
ALPHA="${ALPHA:-0.0}"
LOAD_BALANCE_BETA="${LOAD_BALANCE_BETA:-0.0}"

if [[ -z "$CHECKPOINT_DIR" ]]; then
  if [[ "$MODE" == "local" ]]; then
    CHECKPOINT_DIR="checkpoints"
  else
    CHECKPOINT_DIR="../artifacts/checkpoints"
  fi
fi

cd "$REPO_ROOT"

build_args() {
  local alpha_slug beta_slug exp_name
  alpha_slug="${ALPHA//./p}"
  beta_slug="${LOAD_BALANCE_BETA//./p}"
  exp_name="moe_joint_v0_a${alpha_slug}_b${beta_slug}_s67"

  local args=(
    "experiment=exp06_moe_joint_v0"
    "runtime.device=$RUNTIME_DEVICE"
    "training.checkpoint_dir=$CHECKPOINT_DIR"
    "validation.data_dir=$DATA_ROOT/val"
    "experiment.exp_name=$exp_name"
    "model.compute.flops_penalty_alpha=$ALPHA"
    "model.compute.load_balance_beta=$LOAD_BALANCE_BETA"
  )

  printf '%s ' "${args[@]}"
}

case "$MODE" in
  print)
    echo "uv run python main.py $(build_args)"
    ;;
  local)
    # shellcheck disable=SC2206
    run_args=( $(build_args) )
    uv run python main.py "${run_args[@]}"
    ;;
  qsub)
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
