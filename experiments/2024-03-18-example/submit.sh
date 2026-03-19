#!/usr/bin/env bash
#
# Submit all experiments in this batch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

experiments=(exp01_baseline exp02_moe)

echo "Submitting: Example Batch"
echo "=========================="

for exp in "${experiments[@]}"; do
    echo "Submitting: $exp"
    
    job_id=$(qsub \
        -N "moe-$exp" \
        -v "RUN_ARGS=experiment=$exp" \
        -v "BATCH_NAME=example-batch" \
        -v "EXP_NAME=$exp" \
        "$REPO_ROOT/scripts/metacentrum_job.sh" 2>&1 | grep -oP '^\d+' || echo "FAILED")
    
    if [ "$job_id" != "FAILED" ]; then
        echo "  ✓ Job ID: $job_id"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - $exp - $job_id" >> "$SCRIPT_DIR/submission_log.txt"
    else
        echo "  ✗ Failed"
    fi
    echo ""
done

echo "Done. Check: qstat -u \$USER"
