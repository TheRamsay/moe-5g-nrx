#!/usr/bin/env bash
# Run scripts/evaluate.py on key checkpoints with profiles=[uma, tdlc, deepmimo].
# Stage 1 (DeepMIMO dataset generation) must already have completed before
# running this — check that data/test/deepmimov3/asu_campus1/ exists OR the
# W&B artifact `dataset-test-deepmimov3_asu_campus1:latest` is published.
#
# Modes: print | local | qsub

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="${1:-print}"
DATA_ROOT="${DATA_ROOT:-$HOME/moe-5g-datasets/dense-v1}"
SCENARIO="${SCENARIO:-asu_campus1}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda}"
WALLTIME="${WALLTIME:-01:00:00}"
SELECT_RESOURCES="${SELECT_RESOURCES:-select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb}"

# Profiles: uma+tdlc resolve via local data_dir/.pt; SCENARIO resolves via W&B
# artifact `dataset-test-${SCENARIO}:latest` (auto-downloaded by evaluate.py).

# (label, eval-preset, checkpoint_artifact)
# eval-preset YAMLs bake in profiles=[uma,tdlc,asu_campus1] (avoids the
# qsub-RUN_ARGS bracket-eats-comma trap).
RUNS=(
  "dense_large|eval40_ood_dense_large|knn_moe-5g-nrx/moe-5g-nrx/model-dense_large_final20k_constant_lr_s67-55l1dpby:best"
  "exp26_a2e3|eval41_ood_exp26_a2e3|knn_moe-5g-nrx/moe-5g-nrx/model-moe_alphasweep_asym_a2e3_s67-t6lkdep2:best"
  "exp31_2expert|eval42_ood_exp31_2expert|knn_moe-5g-nrx/moe-5g-nrx/model-moe_ablation_2expert_a2e3_s67-5c0kshem:best"
)

cd "$REPO_ROOT"

echo "Study folder: $SCRIPT_DIR"
echo "Mode: $MODE"
echo "Data root (uma/tdlc local): $DATA_ROOT"
echo "Scenario (W&B artifact): $SCENARIO"
echo

build_eval_args() {
  local exp="$1" ckpt="$2"
  # NOTE: qsub -v RUN_ARGS=... uses commas as variable separators, so passing
  # `evaluation.profiles=[uma,tdlc,$SCENARIO]` here breaks. We bake the
  # profiles list into the experiment YAML preset instead (which Hydra reads
  # from disk, not the CLI) — see TODO when this script is actually run.
  local args=(
    "experiment=$exp"
    "evaluation.checkpoint_artifact=$ckpt"
    "evaluation.checkpoint=null"
    "evaluation.data_dir=$DATA_ROOT/test"
    "validation.data_dir=$DATA_ROOT/val"
    "runtime.device=$RUNTIME_DEVICE"
  )
  printf '%s ' "${args[@]}"
}

case "$MODE" in
  print)
    for entry in "${RUNS[@]}"; do
      label="${entry%%|*}"; rest="${entry#*|}"
      exp="${rest%%|*}"; ckpt="${rest#*|}"
      echo "# $label"
      echo "uv run python scripts/evaluate.py $(build_eval_args "$exp" "$ckpt")"
    done
    ;;
  local)
    for entry in "${RUNS[@]}"; do
      label="${entry%%|*}"; rest="${entry#*|}"
      exp="${rest%%|*}"; ckpt="${rest#*|}"
      echo ">>> Eval $label"
      # shellcheck disable=SC2046
      uv run python scripts/evaluate.py $(build_eval_args "$exp" "$ckpt")
    done
    ;;
  qsub)
    for entry in "${RUNS[@]}"; do
      label="${entry%%|*}"; rest="${entry#*|}"
      exp="${rest%%|*}"; ckpt="${rest#*|}"
      echo ">>> Submitting eval for $label"
      qsub -l "walltime=$WALLTIME" -l "$SELECT_RESOURCES" \
        -v "ENTRYPOINT=scripts/evaluate.py,RUN_ARGS=$(build_eval_args "$exp" "$ckpt")" \
        scripts/metacentrum_job.sh
    done
    ;;
  *)
    echo "Unknown mode: $MODE"; exit 1 ;;
esac
