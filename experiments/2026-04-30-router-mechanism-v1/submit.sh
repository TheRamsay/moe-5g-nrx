#!/usr/bin/env bash
#PBS -N router-mechanism
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb
#PBS -l walltime=01:00:00
#PBS -j oe

# Mechanistic interpretability bundle: A) linear probing, C) per-expert
# specialization, F) decision boundary on PCA plane.
# Output: 3 figures + 1 JSON in docs/figures/.

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="scripts/analyze_router_mechanism.py"
export RUN_ARGS="--checkpoint knn_moe-5g-nrx/moe-5g-nrx/model-moe_alphasweep_asym_a2e3_s67-t6lkdep2:best --data-dir /storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test --max-samples 4000 --device cuda --out docs/figures"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
