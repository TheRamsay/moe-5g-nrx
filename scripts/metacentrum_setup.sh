#!/usr/bin/env bash

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

log() {
    printf '[metacentrum-setup] %s\n' "$*"
}

die() {
    printf '[metacentrum-setup] ERROR: %s\n' "$*" >&2
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
    fi
}

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
    die "repo root not found from $REPO_ROOT"
fi

UV_BIN="${UV_BIN:-}"
if [[ -z "$UV_BIN" ]]; then
    UV_BIN="$(command -v uv || true)"
fi

if [[ -z "$UV_BIN" ]]; then
    log "uv not found; installing into ~/.local/bin"
    export UV_INSTALL_DIR="${UV_INSTALL_DIR:-$HOME/.local/bin}"
    mkdir -p "$UV_INSTALL_DIR"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
    UV_BIN="$UV_INSTALL_DIR/uv"
fi

[[ -x "$UV_BIN" ]] || die "uv binary is not executable: $UV_BIN"

export PATH="$(dirname -- "$UV_BIN"):$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$HOME/.local/share/uv/python}"
export UV_LINK_MODE=copy

PROJECT_STATE_DIR="${PROJECT_STATE_DIR:-$REPO_ROOT/.metacentrum}"
PROJECT_ENV_DIR="${PROJECT_ENV_DIR:-$PROJECT_STATE_DIR/venv}"

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PROJECT_STATE_DIR"

init_modules

log "repo root: $REPO_ROOT"
log "uv binary: $UV_BIN"
log "uv cache: $UV_CACHE_DIR"
log "python installs: $UV_PYTHON_INSTALL_DIR"
log "project env: $PROJECT_ENV_DIR"

if command -v module >/dev/null 2>&1; then
    log "loaded modules after reset: $(module -t list 2>&1 | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
fi

log "installing Python 3.10"
"$UV_BIN" python install 3.10

log "syncing locked runtime environment"
(
    cd "$REPO_ROOT"
    UV_PROJECT_ENVIRONMENT="$PROJECT_ENV_DIR" \
        "$UV_BIN" sync --python 3.10 --frozen --no-dev
)

log "running smoke test"
(
    cd "$REPO_ROOT"
    UV_PROJECT_ENVIRONMENT="$PROJECT_ENV_DIR" \
        "$UV_BIN" run --python 3.10 python - <<'PY'
import hydra
import tensorflow as tf
import torch

print(f"hydra={hydra.__version__}")
print(f"tensorflow={tf.__version__}")
print(f"torch={torch.__version__}")
PY
)

log "environment ready"
du -sh "$PROJECT_ENV_DIR" "$UV_CACHE_DIR"

cat <<EOF

Setup complete.

Recommended usage on MetaCentrum:
  1. Keep the shared uv cache in: $UV_CACHE_DIR
  2. Keep one persistent project env in: $PROJECT_ENV_DIR
  3. Run compute jobs from scratch using scripts/metacentrum_job.sh

This layout avoids repeated downloads while keeping heavy runtime I/O off the frontend.
Remember that home/storage quotas apply to both data volume and file count.

EOF
