#!/usr/bin/env bash
#
# Template for submitting experiment batches
# Usage: Copy to your experiment folder and edit experiments.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

# Read experiments from YAML (simple key=value format)
# Format in experiments.yaml:
#   exp1:
#     model_family: moe
#     router_temperature: 0.3
#   exp2:
#     model_family: moe
#     router_temperature: 1.0

echo "Submitting experiments from $SCRIPT_DIR"
echo ""

# TODO: Parse experiments.yaml and submit each one
# Example:
# qsub -v "RUN_ARGS=model.family=moe model.router.temperature=0.3" \
#      scripts/metacentrum_job.sh

echo "Edit this script to read your experiments.yaml and submit jobs"
