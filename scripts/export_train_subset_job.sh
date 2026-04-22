#!/usr/bin/env bash
#PBS -N array3d-export
#PBS -l select=1:ncpus=4:mem=32gb:scratch_local=10gb
#PBS -l walltime=3:00:00
#PBS -j oe

# Exports a 50k-sample Array3D training subset to persistent storage.
# Output: /storage/brno2/home/ramsay/moe-5g-datasets/train-50k-array3d/{uma,tdlc}
#
# Submit from repo root:
#   qsub scripts/export_train_subset_job.sh

set -Eeuo pipefail
IFS=$'\n\t'

log() { printf '[array3d-export][%s] %s\n' "${PBS_JOBID:-local}" "$*"; }
die() { printf '[array3d-export][%s] ERROR: %s\n' "${PBS_JOBID:-local}" "$*" >&2; exit 1; }

REPO_ROOT="${REPO_ROOT:-${PBS_O_WORKDIR:-}}"
[[ -n "$REPO_ROOT" ]] || die "submit from repo root or set REPO_ROOT"
[[ -f "$REPO_ROOT/pyproject.toml" ]] || die "pyproject.toml not found under $REPO_ROOT"
[[ -n "${SCRATCHDIR:-}" ]] || die "SCRATCHDIR not set"

SUBMIT_HOME="${PBS_O_HOME:-$HOME}"
if [[ -z "$SUBMIT_HOME" || ! -d "$SUBMIT_HOME" || "$SUBMIT_HOME" == /scratch* ]]; then
    SUBMIT_HOME="$(getent passwd "${USER:-$(id -un)}" | cut -d: -f6 2>/dev/null || true)"
fi
export HOME="$SUBMIT_HOME"

UV_BIN="${UV_BIN:-$SUBMIT_HOME/.local/bin/uv}"
[[ -x "$UV_BIN" ]] || UV_BIN="$(command -v uv || true)"
[[ -n "$UV_BIN" ]] || die "uv not found; run scripts/metacentrum_setup.sh first"

export PATH="$(dirname -- "$UV_BIN"):$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SUBMIT_HOME/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$SUBMIT_HOME/.local/share/uv/python}"
export UV_LINK_MODE=copy

export TMPDIR="$SCRATCHDIR/tmp"
export XDG_CACHE_HOME="$SCRATCHDIR/.cache"
export MPLCONFIGDIR="$SCRATCHDIR/.config/matplotlib"
export PYTHONUNBUFFERED=1

# HF cache on persistent storage — same logic as metacentrum_job.sh
unset HF_HUB_CACHE HF_DATASETS_CACHE
export HF_HOME="$SUBMIT_HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

WORK_ROOT="$SCRATCHDIR/work"

log "staging repository to scratch"
mkdir -p "$WORK_ROOT"
rsync -a \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude 'results/' \
    "$REPO_ROOT/" "$WORK_ROOT/"

cd "$WORK_ROOT"

log "creating environment"
UV_PROJECT_ENVIRONMENT="$WORK_ROOT/.venv" \
    "$UV_BIN" sync --python 3.10 --frozen --offline --no-dev

OUTPUT_DIR="$SUBMIT_HOME/moe-5g-datasets/train-50k-array3d"
log "output directory: $OUTPUT_DIR"

log "running export"
UV_PROJECT_ENVIRONMENT="$WORK_ROOT/.venv" \
    "$UV_BIN" run --offline --python 3.10 python scripts/export_train_subset.py \
        --output "$OUTPUT_DIR" \
        --n_samples 50000 \
        --profiles uma tdlc

log "done — dataset at: $OUTPUT_DIR"
log "use in training jobs:"
log "  training.hf_train_data_dir=$OUTPUT_DIR"
log "  training.hf_max_samples=50000"
