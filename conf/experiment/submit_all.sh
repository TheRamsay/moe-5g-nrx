#!/usr/bin/env bash
#
# Submit all experiments in the suite
# Usage: bash submit_all.sh

set -euo pipefail

echo "Submitting experiment suite: MoE Router Variants"
echo "================================================="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

# Define experiments
experiments=(
    "exp01_baseline_dense:Baseline Dense"
    "exp02_moe_default:MoE Default"
    "exp03_moe_hard_routing:MoE Hard Routing"
    "exp04_moe_soft_routing:MoE Soft Routing"
    "exp05_moe_high_penalty:MoE High Penalty"
)

# Submit each experiment
for exp in "${experiments[@]}"; do
    config="${exp%%:*}"
    name="${exp##*:}"
    
    echo "Submitting: $name"
    
    job_id=$(qsub \
        -N "moe-exp-${config}" \
        -v "RUN_ARGS=experiment=$config" \
        "$REPO_ROOT/scripts/metacentrum_job.sh" 2>&1 | grep -oP '^\d+' || echo "FAILED")
    
    if [ "$job_id" != "FAILED" ]; then
        echo "  ✓ Submitted with Job ID: $job_id"
        # Log to tracking file
        echo "$(date '+%Y-%m-%d %H:%M:%S') - $config - $job_id" >> "$SCRIPT_DIR/submission_log.txt"
    else
        echo "  ✗ Failed to submit"
    fi
    echo ""
done

echo "================================================="
echo "All experiments submitted!"
echo ""
echo "Check status: qstat -u \$USER"
echo "View results: https://wandb.ai/[entity]/moe-5g-nrx"
echo ""
echo "Submission log saved to: conf/experiment/submission_log.txt"
