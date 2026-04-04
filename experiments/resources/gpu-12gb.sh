#!/usr/bin/env bash

# Older ~12 GB GPU class. Fine for smoke tests, not the preferred default.
export RESOURCE_PRESET_NAME="gpu-12gb"
export SELECT_RESOURCES="select=1:ncpus=4:ngpus=1:mem=24gb:scratch_ssd=40gb:gpu_mem=12288mb"
export WALLTIME="${WALLTIME:-04:00:00}"

printf 'Loaded resource preset: %s\n' "$RESOURCE_PRESET_NAME"
printf '  SELECT_RESOURCES=%s\n' "$SELECT_RESOURCES"
printf '  WALLTIME=%s\n' "$WALLTIME"
