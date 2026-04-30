#!/usr/bin/env bash
#PBS -N export-100k-subset
#PBS -l select=1:ncpus=4:mem=96gb:scratch_ssd=120gb
#PBS -l walltime=04:00:00
#PBS -j oe

# One-time export of 100k training samples per profile to persistent storage.
# Output: /storage/brno2/home/ramsay/moe-5g-datasets/train-100k-array3d/{uma,tdlc}
# Reads from HF cache (already on cluster: 124 GB).
# Memory budget: 96 GB to safely hold the 100k cast operation.

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="scripts/export_train_subset.py"
export RUN_ARGS="--n_samples 100000 --output /storage/brno2/home/ramsay/moe-5g-datasets/train-100k-array3d"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
