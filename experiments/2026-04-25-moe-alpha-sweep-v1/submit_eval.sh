#!/usr/bin/env bash
# Run scripts/evaluate.py on each :best checkpoint from the alpha sweep.
# Spawns 4 fresh wandb eval runs (named *_eval_uma-tdlc) with full val + test
# BLER, FLOPs, routing tables — what the Pareto curve needs.
#
# Why local-checkpoint-only:
# - exp24 (job 19457669) and exp27 (job 19457672) had wandb-init flake during
#   training — checkpoints saved cleanly but not uploaded to W&B as artifacts.
# - exp25 + exp26 did upload :best artifacts to W&B, but using the local sync
#   keeps the same code path for all 4 (and avoids a download).
#
# Modes: print | local | qsub

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="${1:-print}"
DATA_ROOT="${DATA_ROOT:-$HOME/moe-5g-datasets/dense-v1}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda}"
WALLTIME="${WALLTIME:-01:00:00}"
SELECT_RESOURCES="${SELECT_RESOURCES:-select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb}"

# (experiment_preset, training_job_id) — checkpoint lives at
# $RESULTS_ROOT/<job_id>.pbs-m1.metacentrum.cz/checkpoints/compute_aware_moe_nrx_best.pt
RUNS=(
  "exp24_moe_alphasweep_a5e4:19457669"
  "exp25_moe_alphasweep_a1e3:19457670"
  "exp26_moe_alphasweep_a2e3:19457671"
  "exp27_moe_alphasweep_a5e3:19457672"
)

cd "$REPO_ROOT"

echo "Study folder: $SCRIPT_DIR"
echo "Mode: $MODE"
echo "Data root (val/test): $DATA_ROOT"
echo "Results root: $RESULTS_ROOT"
echo

build_eval_args() {
  local exp="$1" job_id="$2"
  local ckpt="$RESULTS_ROOT/${job_id}.pbs-m1.metacentrum.cz/checkpoints/compute_aware_moe_nrx_best.pt"
  printf '%s ' \
    "experiment=$exp" \
    "evaluation.checkpoint=$ckpt" \
    "evaluation.checkpoint_artifact=null" \
    "evaluation.data_dir=$DATA_ROOT/test" \
    "validation.data_dir=$DATA_ROOT/val" \
    "runtime.device=$RUNTIME_DEVICE"
}

case "$MODE" in
  print)
    for entry in "${RUNS[@]}"; do
      exp="${entry%%:*}"
      job_id="${entry##*:}"
      echo "uv run python scripts/evaluate.py $(build_eval_args "$exp" "$job_id")"
    done
    ;;
  local)
    for entry in "${RUNS[@]}"; do
      exp="${entry%%:*}"
      job_id="${entry##*:}"
      echo ">>> Eval $exp (ckpt from $job_id)"
      # shellcheck disable=SC2046
      uv run python scripts/evaluate.py $(build_eval_args "$exp" "$job_id")
    done
    ;;
  qsub)
    # 4 separate jobs — metacentrum_job.sh runs one python ENTRYPOINT per job.
    # Each eval is <5min so 4-in-the-queue is cheap; the wandb-init flake hits
    # less often when jobs land on different compute hosts (which 4 jobs do).
    for entry in "${RUNS[@]}"; do
      exp="${entry%%:*}"
      job_id="${entry##*:}"
      echo ">>> Submitting eval for $exp (ckpt from $job_id)"
      qsub -l "walltime=$WALLTIME" -l "$SELECT_RESOURCES" \
        -v "ENTRYPOINT=scripts/evaluate.py,RUN_ARGS=$(build_eval_args "$exp" "$job_id")" \
        scripts/metacentrum_job.sh
    done
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash submit_eval.sh [print|local|qsub]"
    exit 1
    ;;
esac
