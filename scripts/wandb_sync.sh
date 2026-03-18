#!/usr/bin/env bash
#
# Sync offline wandb runs to cloud from MetaCentrum frontend
# Usage: ./scripts/wandb_sync.sh [results_directory]
#

set -euo pipefail

RESULTS_DIR="${1:-./results}"

echo "Scanning for wandb offline runs in: $RESULTS_DIR"
echo ""

if ! command -v wandb &>/dev/null; then
    echo "ERROR: wandb CLI not found"
    echo "Install with: uv pip install wandb"
    exit 1
fi

# Check internet connectivity
if ! curl -s --max-time 5 https://api.wandb.ai/healthz &>/dev/null; then
    echo "ERROR: Cannot reach wandb servers. Are you on a frontend with internet?"
    exit 1
fi

sync_count=0

# Find all wandb directories in results
while IFS= read -r -d '' wandb_dir; do
    echo "Found wandb directory: $wandb_dir"
    
    # Check for offline runs
    if [[ -d "$wandb_dir/offline-run-"* ]]; then
        echo "  → Syncing offline runs..."
        if wandb sync "$wandb_dir" --sync-all; then
            echo "  ✓ Synced successfully"
            ((sync_count++))
        else
            echo "  ✗ Sync failed"
        fi
    elif [[ -d "$wandb_dir/run-"* ]]; then
        echo "  → Online runs found (already synced)"
    else
        echo "  → No runs to sync"
    fi
    echo ""
done < <(find "$RESULTS_DIR" -type d -name "wandb" -print0 2>/dev/null || true)

if [[ $sync_count -eq 0 ]]; then
    echo "No offline wandb runs found to sync"
    echo ""
    echo "Tip: Make sure your jobs ran with logging.use_wandb=true"
    exit 0
else
    echo "Synced $sync_count wandb run(s)"
    echo ""
    echo "View at: https://wandb.ai/$(wandb login 2>&1 | grep -oP 'entity: \K\S+' || echo 'your-entity')"
fi
