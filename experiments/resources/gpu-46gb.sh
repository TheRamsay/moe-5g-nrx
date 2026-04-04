#!/usr/bin/env bash

# Larger 46 GB class for heavier jobs or extra headroom.
export RESOURCE_PRESET_NAME="gpu-46gb"
export SELECT_RESOURCES="select=1:ncpus=4:ngpus=1:mem=32gb:scratch_ssd=40gb:gpu_mem=46068mb"
export WALLTIME="${WALLTIME:-08:00:00}"

printf 'Loaded resource preset: %s\n' "$RESOURCE_PRESET_NAME"
printf '  SELECT_RESOURCES=%s\n' "$SELECT_RESOURCES"
printf '  WALLTIME=%s\n' "$WALLTIME"
