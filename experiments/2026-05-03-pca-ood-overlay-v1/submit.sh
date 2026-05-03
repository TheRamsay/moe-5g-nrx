#!/usr/bin/env bash
#PBS -N pca-ood-overlay
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb
#PBS -l walltime=01:00:00
#PBS -j oe

# PCA-2D overlay of in-distribution (UMa+TDLC) vs OOD (ASU ray-traced) stem features.
# Visual evidence that OOD samples land outside the in-dist feature manifold.
# Output: docs/figures/pca_ood_overlay.{pdf,png,npz}

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="scripts/visualize_ood_pca.py"
export RUN_ARGS="--checkpoint knn_moe-5g-nrx/moe-5g-nrx/model-moe_alphasweep_asym_a2e3_s67-t6lkdep2:best --max-samples 2000 --device cuda --out /storage/brno2/home/ramsay/moe-5g-nrx/docs/figures"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
