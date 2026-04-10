#!/usr/bin/env bash
#PBS -N hf-predownload
#PBS -l select=1:ncpus=2:mem=8gb:scratch_ssd=10gb
#PBS -l walltime=04:00:00
#PBS -j oe

# Pre-download the HF dataset to persistent storage so training jobs start instantly.
# CPU-only, no GPU — this is just network + disk I/O.
#
# Submit with:
#   qsub scripts/predownload_hf.sh
#
# Override the dataset with:
#   qsub -v HF_DATASET=Other/dataset scripts/predownload_hf.sh

set -Eeuo pipefail
IFS=$'\n\t'

HF_DATASET="${HF_DATASET:-Vack0/moe-5g-nrx}"
HF_CONFIGS="${HF_CONFIGS:-uma tdlc}"
HF_SPLITS="${HF_SPLITS:-train val test}"

REPO_ROOT="${REPO_ROOT:-${PBS_O_WORKDIR:-}}"
[[ -n "$REPO_ROOT" ]] || { echo "ERROR: REPO_ROOT not set"; exit 1; }

SUBMIT_HOME="${PBS_O_HOME:-$HOME}"
UV_BIN="${UV_BIN:-$SUBMIT_HOME/.local/bin/uv}"
[[ -x "$UV_BIN" ]] || UV_BIN="$(command -v uv)"

# Force HF caches to persistent storage. MetaCentrum pre-sets HF_HUB_CACHE
# and HF_DATASETS_CACHE to $SCRATCHDIR, so unset them first before reassigning.
unset HF_HUB_CACHE HF_DATASETS_CACHE
export HF_HOME="$SUBMIT_HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$SUBMIT_HOME/.cache/uv}"
export TMPDIR="$SCRATCHDIR/tmp"
mkdir -p "$TMPDIR"

cd "$REPO_ROOT"

echo "[predownload] HF_HOME=$HF_HOME"
echo "[predownload] HF_HUB_CACHE=$HF_HUB_CACHE"
echo "[predownload] HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "[predownload] Dataset: $HF_DATASET"
echo "[predownload] Configs: $HF_CONFIGS"
echo "[predownload] Splits: $HF_SPLITS"
echo

"$UV_BIN" run --offline --python 3.10 python - <<PY
import os
from datasets import load_dataset

dataset = "$HF_DATASET"
configs = "$HF_CONFIGS".split()
splits = "$HF_SPLITS".split()

for config in configs:
    for split in splits:
        print(f"[predownload] loading {dataset}/{config}/{split}...", flush=True)
        ds = load_dataset(dataset, config, split=split)
        print(f"[predownload]   -> {len(ds)} samples", flush=True)

print("[predownload] done", flush=True)
PY

echo "[predownload] job finished"

cleanup() {
    if command -v clean_scratch >/dev/null 2>&1; then
        clean_scratch || true
    fi
}
trap cleanup EXIT
