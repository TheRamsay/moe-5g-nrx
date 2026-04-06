#!/usr/bin/env bash
# Waterfall BLER comparison: nano vs small vs large_s32
# Proves (or disproves) that expert size matters in the waterfall region.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="${1:-print}"
DATA_ROOT="${DATA_ROOT:-$HOME/moe-5g-datasets/dense-v1}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda}"
WALLTIME="${WALLTIME:-02:00:00}"

# Artifact references for the three checkpoints
NANO_ARTIFACT="knn_moe-5g-nrx/moe-5g-nrx/model-dense_nano_final20k_constant_lr_s67-aos4hhid:best"
SMALL_ARTIFACT="knn_moe-5g-nrx/moe-5g-nrx/model-dense_small_final20k_constant_lr_s67-kivdz4qu:best"
LARGE_S32_ARTIFACT="knn_moe-5g-nrx/moe-5g-nrx/model-dense_large_s32_final20k_constant_lr_s67-rdfefyt1:best"

# Also compare the original large (s56) for reference
LARGE_S56_ARTIFACT="knn_moe-5g-nrx/moe-5g-nrx/model-dense_large_final20k_constant_lr_s67-55l1dpby:best"

cd "$REPO_ROOT"

source experiments/resources/gpu-16gb.sh

echo "Study: waterfall-compare-v1"
echo "Mode: $MODE"
echo "Data root: $DATA_ROOT"
echo

ENTRYPOINT="scripts/waterfall_compare.py"

# On MetaCentrum, ../artifacts/ gets synced back to results/
if [[ "$MODE" == "qsub" ]]; then
  OUTPUT_DIR="../artifacts"
else
  OUTPUT_DIR="experiments/2026-04-06-waterfall-compare-v1"
fi

RUN_ARGS="--artifacts $NANO_ARTIFACT $SMALL_ARTIFACT $LARGE_S32_ARTIFACT $LARGE_S56_ARTIFACT --labels nano small large_s32 large_s56 --data-dir $DATA_ROOT/val --profiles tdlc uma --snr-step 2 --device $RUNTIME_DEVICE --output $OUTPUT_DIR/waterfall.png"

case "$MODE" in
  print)
    echo "uv run python $ENTRYPOINT $RUN_ARGS"
    ;;
  local)
    echo ">>> Running waterfall comparison locally"
    uv run python $ENTRYPOINT $RUN_ARGS
    ;;
  qsub)
    echo ">>> Submitting waterfall comparison"
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
