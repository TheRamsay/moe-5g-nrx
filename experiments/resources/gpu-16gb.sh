#!/usr/bin/env bash

# Recommended default for dense runs. Matches the successful 16 GB class smoke/baseline runs.
export RESOURCE_PRESET_NAME="gpu-16gb"
export SELECT_RESOURCES="select=1:ncpus=4:ngpus=1:mem=24gb:scratch_ssd=40gb:gpu_mem=16384mb"
export WALLTIME="${WALLTIME:-08:00:00}"

printf 'Loaded resource preset: %s\n' "$RESOURCE_PRESET_NAME"
printf '  SELECT_RESOURCES=%s\n' "$SELECT_RESOURCES"
printf '  WALLTIME=%s\n' "$WALLTIME"
