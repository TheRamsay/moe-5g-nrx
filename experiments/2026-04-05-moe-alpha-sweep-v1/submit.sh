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
ALPHA="${ALPHA:-1e-3}"
LOAD_BALANCE_BETA="${LOAD_BALANCE_BETA:-0.1}"

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
  alpha_slug="${alpha_slug//-/_}"
  beta_slug="${LOAD_BALANCE_BETA//./p}"
  exp_name="moe_alpha_v1_a${alpha_slug}_b${beta_slug}_s67"

  local args=(
    "experiment=exp06_moe_joint_v0"
    "runtime.device=$RUNTIME_DEVICE"
    "training.checkpoint_dir=$CHECKPOINT_DIR"
    "validation.data_dir=$DATA_ROOT/val"
    "experiment.exp_name=$exp_name"
    "experiment.batch_name=moe-alpha-sweep-v1"
    "experiment.study_slug=2026-04-05-moe-alpha-sweep-v1"
    "experiment.study_path=experiments/2026-04-05-moe-alpha-sweep-v1"
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
