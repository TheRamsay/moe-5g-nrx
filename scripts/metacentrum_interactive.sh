#!/usr/bin/env bash
#
# Interactive job helper for MetaCentrum
# Usage: ./scripts/metacentrum_interactive.sh [hours]
#
# Once connected, run: source scripts/interactive_env.sh
#

set -euo pipefail

HOURS="${1:-2}"

echo "Requesting interactive GPU job for ${HOURS} hours..."
echo ""
echo "Once connected, run: source scripts/interactive_env.sh"
echo ""

qsub -I \
    -l "select=1:ncpus=4:ngpus=1:mem=32gb:scratch_ssd=40gb" \
    -l "walltime=${HOURS}:00:00"
