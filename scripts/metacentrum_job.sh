#!/usr/bin/env bash
#PBS -N moe-5g-nrx
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:scratch_ssd=40gb
#PBS -l walltime=04:00:00
#PBS -j oe

set -Eeuo pipefail
IFS=$'\n\t'

DEFAULT_GPU_MODULES="${DEFAULT_GPU_MODULES:-cuda/11.6.2-gcc-10.2.1-nwpmxyy cudnn/8.4.0.27-11.6-gcc-10.2.1-pqxrvlk}"

add_modules() {
    local modules_string="$1"
    local -a modules=()
    local old_ifs="$IFS"
    IFS=' '
    read -r -a modules <<< "$modules_string"
    IFS="$old_ifs"
    if [[ ${#modules[@]} -gt 0 ]]; then
        module add "${modules[@]}"
    fi
}

split_cli_args() {
    local args_string="$1"
    local -n out_array_ref="$2"
    local old_ifs="$IFS"
    IFS=' '
    read -r -a out_array_ref <<< "$args_string"
    IFS="$old_ifs"
}

log() {
    printf '[metacentrum-job][%s] %s\n' "${PBS_JOBID:-local}" "$*"
}

die() {
    printf '[metacentrum-job][%s] ERROR: %s\n' "${PBS_JOBID:-local}" "$*" >&2
    exit 1
}

init_modules() {
    if ! command -v module >/dev/null 2>&1; then
        [[ -r /etc/profile ]] && source /etc/profile
    fi

    if ! command -v module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh
    fi

    if command -v module >/dev/null 2>&1; then
        module purge >/dev/null 2>&1 || true
        module add metabase/1 >/dev/null 2>&1 || true

        if [[ -n "$DEFAULT_GPU_MODULES" ]]; then
            add_modules "$DEFAULT_GPU_MODULES"
        fi

        if [[ -n "${EXTRA_MODULES:-}" ]]; then
            add_modules "$EXTRA_MODULES"
        fi
    fi
}

REPO_ROOT="${REPO_ROOT:-${PBS_O_WORKDIR:-}}"
[[ -n "$REPO_ROOT" ]] || die "set REPO_ROOT or submit the job from the repo root"
[[ -f "$REPO_ROOT/pyproject.toml" ]] || die "pyproject.toml not found under $REPO_ROOT"
[[ -n "${SCRATCHDIR:-}" ]] || die "SCRATCHDIR is not set; request scratch_local in qsub"
[[ -d "$SCRATCHDIR" ]] || die "SCRATCHDIR does not exist: $SCRATCHDIR"

SUBMIT_HOME="${PBS_O_HOME:-$HOME}"
UV_BIN="${UV_BIN:-$SUBMIT_HOME/.local/bin/uv}"
if [[ ! -x "$UV_BIN" ]]; then
    UV_BIN="$(command -v uv || true)"
fi
[[ -n "$UV_BIN" ]] || die "uv not found; run scripts/metacentrum_setup.sh first"

init_modules

export PATH="$(dirname -- "$UV_BIN"):$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SUBMIT_HOME/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$SUBMIT_HOME/.local/share/uv/python}"
export UV_LINK_MODE=copy

WORK_ROOT="$SCRATCHDIR/work"
ARTIFACT_DIR="$SCRATCHDIR/artifacts"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results}"
JOB_LABEL="${PBS_JOBID:-$(date +%Y%m%d-%H%M%S)}"
JOB_RESULT_DIR="$RESULTS_ROOT/$JOB_LABEL"

export TMPDIR="$SCRATCHDIR/tmp"
export XDG_CACHE_HOME="$SCRATCHDIR/.cache"
export MPLCONFIGDIR="$SCRATCHDIR/.config/matplotlib"
export WANDB_DIR="$ARTIFACT_DIR/wandb"
export WANDB_CACHE_DIR="$SCRATCHDIR/.cache/wandb"
export RUN_OUTPUT_DIR="$ARTIFACT_DIR"
export PYTHONUNBUFFERED=1

mkdir -p "$WORK_ROOT" "$ARTIFACT_DIR" "$JOB_RESULT_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$WANDB_DIR"

cleanup() {
    local status=$?
    set +e

    if [[ -d "$ARTIFACT_DIR" ]]; then
        rsync -a "$ARTIFACT_DIR/" "$JOB_RESULT_DIR/"
    fi

    if command -v clean_scratch >/dev/null 2>&1; then
        clean_scratch || true
    else
        find "$SCRATCHDIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + || true
    fi

    exit "$status"
}

trap cleanup EXIT

log "staging repository to scratch"
rsync -a \
    --delete \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '.metacentrum/' \
    --exclude '.ruff_cache/' \
    --exclude '__pycache__/' \
    --exclude 'results/' \
    --exclude 'dist/' \
    --exclude 'outputs/' \
    "$REPO_ROOT/" "$WORK_ROOT/"

cd "$WORK_ROOT"

log "creating scratch-local environment from the shared uv cache"
UV_PROJECT_ENVIRONMENT="$WORK_ROOT/.venv" \
    "$UV_BIN" sync --python 3.10 --frozen --offline --no-dev

printf 'job_id=%s\n' "${PBS_JOBID:-local}" > "$ARTIFACT_DIR/job.txt"
printf 'hostname=%s\n' "$(hostname -f)" >> "$ARTIFACT_DIR/job.txt"
printf 'scratchdir=%s\n' "$SCRATCHDIR" >> "$ARTIFACT_DIR/job.txt"
printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-}" >> "$ARTIFACT_DIR/job.txt"

if command -v module >/dev/null 2>&1; then
    module -t list > "$ARTIFACT_DIR/modules.txt" 2>&1 || true
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "$ARTIFACT_DIR/nvidia-smi.txt" 2>&1 || true
fi

ENTRYPOINT="${ENTRYPOINT:-main.py}"
RUN_ARGS="${RUN_ARGS:-model=static_dense dataset=mixed runtime.device=cuda}"

split_cli_args "$RUN_ARGS" run_args
cmd=("$UV_BIN" run --offline --python 3.10 python "$ENTRYPOINT" "${run_args[@]}")

printf '%q ' "${cmd[@]}" > "$ARTIFACT_DIR/command.sh"
printf '\n' >> "$ARTIFACT_DIR/command.sh"

log "starting workload"
"${cmd[@]}" 2>&1 | tee "$ARTIFACT_DIR/run.log"

log "job finished successfully"
