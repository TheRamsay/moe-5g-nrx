#!/usr/bin/env bash
# Re-eval the 3 dense baselines so per-SNR-bin BLER lands in W&B summary
# (the original evals from 2026-04 used an older evaluate.py that only
# logged the per-SNR data as table artifacts, not flat summary keys).
#
# Modes: print | qsub

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="${1:-print}"
DATA_ROOT="${DATA_ROOT:-$HOME/moe-5g-datasets/dense-v1}"
WALLTIME="${WALLTIME:-00:30:00}"
SELECT_RESOURCES="${SELECT_RESOURCES:-select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb}"

# (label, experiment_preset, checkpoint_artifact)
RUNS=(
  "dense_nano|exp11_dense_nano_final20k|knn_moe-5g-nrx/moe-5g-nrx/model-dense_nano_final20k_constant_lr_s67-aos4hhid:best"
  "dense_small|exp03_dense_capacity_small|knn_moe-5g-nrx/moe-5g-nrx/model-dense_small_final20k_constant_lr_s67-kivdz4qu:best"
  "dense_large|exp16_dense_large_hf_tuned|knn_moe-5g-nrx/moe-5g-nrx/model-dense_large_final20k_constant_lr_s67-55l1dpby:best"
)

cd "$REPO_ROOT"

build_eval_args() {
  local exp="$1" ckpt="$2"
  # NOTE: don't pass evaluation.profiles=[uma,tdlc] — Hydra default already is
  # [uma, tdlc], and brackets in qsub -v RUN_ARGS get mangled by PBS.
  printf '%s ' \
    "experiment=$exp" \
    "evaluation.checkpoint_artifact=$ckpt" \
    "evaluation.checkpoint=null" \
    "evaluation.snr_bins=7" \
    "evaluation.data_dir=$DATA_ROOT/test" \
    "validation.data_dir=$DATA_ROOT/val" \
    "runtime.device=cuda"
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
