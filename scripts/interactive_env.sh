#!/usr/bin/env bash
#
# Source this script in an interactive MetaCentrum job to set up environment
# Usage: source scripts/interactive_env.sh
#

# Guard: must be sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: This script must be sourced, not executed." >&2
    echo "Usage: source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

# Guard: must be in an interactive PBS job
if [[ -z "${SCRATCHDIR:-}" ]]; then
    echo "ERROR: SCRATCHDIR not set. Are you in an interactive PBS job?" >&2
    echo "Start interactive job first:" >&2
    echo "  qsub -I -l select=1:ncpus=4:ngpus=1:mem=32gb:scratch_ssd=40gb -l walltime=2:00:00" >&2
    return 1
fi

set -euo pipefail

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

# =============================================================================
# Configuration
# =============================================================================

REPO_ROOT="${REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"
if [[ ! -x "$UV_BIN" ]]; then
    UV_BIN="$(command -v uv || true)"
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${HOME}/.local/share/uv/python}"
export UV_LINK_MODE=copy

WORK_ROOT="${SCRATCHDIR}/work"
ARTIFACT_DIR="${SCRATCHDIR}/artifacts"

# Temp directories on fast scratch
export TMPDIR="${SCRATCHDIR}/tmp"
export XDG_CACHE_HOME="${SCRATCHDIR}/.cache"
export MPLCONFIGDIR="${SCRATCHDIR}/.config/matplotlib"
export WANDB_DIR="${ARTIFACT_DIR}/wandb"
export WANDB_CACHE_DIR="${SCRATCHDIR}/.cache/wandb"
export RUN_OUTPUT_DIR="${ARTIFACT_DIR}"
export PYTHONUNBUFFERED=1

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    printf '[interactive-env] %s\n' "$*"
}

die() {
    printf '[interactive-env] ERROR: %s\n' "$*" >&2
    return 1
}

# Stage code from home to scratch
stage_code() {
    log "Staging repository to scratch..."
    mkdir -p "$WORK_ROOT"
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
    log "Code staged to: $WORK_ROOT"
}

# Create venv on scratch from cached packages
setup_venv() {
    log "Setting up Python environment on scratch..."
    mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$WANDB_DIR"
    (
        cd "$WORK_ROOT"
        UV_PROJECT_ENVIRONMENT="$WORK_ROOT/.venv" \
            "$UV_BIN" sync --python 3.10 --frozen --offline --no-dev
    )
    log "Environment ready at: $WORK_ROOT/.venv"
}

# Sync results back to home
sync_back() {
    local result_dir="${REPO_ROOT}/results/interactive-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$result_dir"
    if [[ -d "$ARTIFACT_DIR" ]]; then
        rsync -a "$ARTIFACT_DIR/" "$result_dir/"
        log "Results synced to: $result_dir"
    fi
}

# Quick experiment runner
run_experiment() {
    local args="${1:-model=static_dense dataset=mixed runtime.device=cuda}"
    local run_args=()
    log "Running: python main.py $args"
    split_cli_args "$args" run_args
    (
        cd "$WORK_ROOT"
        UV_PROJECT_ENVIRONMENT="$WORK_ROOT/.venv" \
            "$UV_BIN" run --offline --python 3.10 python main.py "${run_args[@]}" 2>&1 | tee -a "$ARTIFACT_DIR/run.log"
    )
}

# Show current environment status
env_status() {
    echo ""
    echo "=== Interactive Environment Status ==="
    echo "Job ID:        ${PBS_JOBID:-N/A}"
    echo "Hostname:      $(hostname -f)"
    echo "Scratch:       $SCRATCHDIR"
    echo "Work dir:      $WORK_ROOT"
    echo "Artifacts:     $ARTIFACT_DIR"
    echo "UV binary:     $UV_BIN"
    echo "UV cache:      $UV_CACHE_DIR"
    echo "Python:        $("$UV_BIN" run --offline --python 3.10 python --version 2>/dev/null || echo 'Not ready')"
    echo "CUDA devices:  ${CUDA_VISIBLE_DEVICES:-N/A}"
    if command -v nvidia-smi &>/dev/null; then
        echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
    fi
    echo ""
    echo "Available commands:"
    echo "  stage_code     - Re-sync code from home to scratch"
    echo "  setup_venv     - Recreate Python environment"
    echo "  run_experiment [args] - Run main.py with Hydra args"
    echo "  sync_back      - Copy results to home directory"
    echo "  env_status     - Show this status"
    echo ""
}

# =============================================================================
# Main Setup
# =============================================================================

log "Setting up interactive environment..."
log "Job ID: ${PBS_JOBID:-unknown}"
log "Scratch: $SCRATCHDIR"

# Check uv is available
if [[ -z "$UV_BIN" ]] || [[ ! -x "$UV_BIN" ]]; then
    die "uv not found. Run ./scripts/metacentrum_setup.sh first on the frontend."
    return 1
fi

# Add uv to PATH
export PATH="$(dirname -- "$UV_BIN"):$PATH"

# Initialize modules
if command -v module &>/dev/null; then
    module purge &>/dev/null || true
    module add metabase/1 &>/dev/null || true
    if [[ -n "$DEFAULT_GPU_MODULES" ]]; then
        add_modules "$DEFAULT_GPU_MODULES" &>/dev/null || true
    fi
    if [[ -n "${EXTRA_MODULES:-}" ]]; then
        add_modules "$EXTRA_MODULES" &>/dev/null || true
    fi
fi

# Stage and setup
stage_code
setup_venv

# Create artifact directory
mkdir -p "$ARTIFACT_DIR"

# Capture job info
printf 'job_id=%s\n' "${PBS_JOBID:-interactive}" > "$ARTIFACT_DIR/job.txt"
printf 'hostname=%s\n' "$(hostname -f)" >> "$ARTIFACT_DIR/job.txt"
printf 'scratchdir=%s\n' "$SCRATCHDIR" >> "$ARTIFACT_DIR/job.txt"
printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-}" >> "$ARTIFACT_DIR/job.txt"

if command -v nvidia-smi &>/dev/null; then
    nvidia-smi > "$ARTIFACT_DIR/nvidia-smi.txt" 2>/dev/null || true
fi

# Change to work directory
cd "$WORK_ROOT"

# Show status
env_status

log "Environment ready! You're in: $WORK_ROOT"
log "Run experiments with: run_experiment 'experiment=exp01_baseline runtime.device=cuda'"
