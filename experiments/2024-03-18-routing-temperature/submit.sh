#!/usr/bin/env bash
#
# Submit all experiments in this batch
# Usage: bash submit.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

echo "Submitting: Router Temperature Sweep"
echo "====================================="
echo ""

# Define experiments (could parse from YAML, but keeping simple)
declare -A experiments=(
    ["exp01_baseline_dense"]="model.family=static_dense training.max_steps=10000"
    ["exp02_moe_default"]="model.family=moe model.router.temperature=1.0 training.max_steps=10000"
    ["exp03_moe_hard_routing"]="model.family=moe model.router.temperature=0.3 training.max_steps=10000"
    ["exp04_moe_soft_routing"]="model.family=moe model.router.temperature=2.0 training.max_steps=10000"
    ["exp05_moe_high_penalty"]="model.family=moe model.compute.flops_penalty_alpha=0.001 training.max_steps=10000"
)

# Submit each
for exp_name in "${!experiments[@]}"; do
    params="${experiments[$exp_name]}"
    
    echo "Submitting: $exp_name"
    echo "  Params: $params"
    
    job_id=$(qsub \
        -N "moe-$exp_name" \
        -v "RUN_ARGS=$params" \
        "$REPO_ROOT/scripts/metacentrum_job.sh" 2>&1 | grep -oP '^\d+' || echo "FAILED")
    
    if [ "$job_id" != "FAILED" ]; then
        echo "  ✓ Job ID: $job_id"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - $exp_name - $job_id" >> "$SCRIPT_DIR/submission_log.txt"
    else
        echo "  ✗ Failed"
    fi
    echo ""
done

echo "====================================="
echo "All submitted! Check: qstat -u \$USER"
echo "View: https://wandb.ai/[entity]/moe-5g-nrx"
