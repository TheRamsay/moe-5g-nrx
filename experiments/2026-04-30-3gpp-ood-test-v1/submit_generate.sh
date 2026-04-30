#!/usr/bin/env bash
#PBS -N gen-3gpp-ood-test
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb
#PBS -l walltime=04:00:00
#PBS -j oe

# Generate OOD test data for TDL-A (low delay spread NLOS), TDL-D (Rician
# LOS-dominant), and CDL-A (clustered NLOS with spatial structure).
# Deterministic seed: base_seed=67 + TEST_SEED_OFFSET (matches existing
# UMa/TDL-C test data so OOD comparisons share the same RNG family).
#
# Output: /storage/.../dense-v1/test/{tdla,tdld,cdla}.pt
# 32k samples per profile. Does NOT touch existing tdlc.pt or uma.pt.

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="scripts/generate_datasets.py"
export RUN_ARGS="generation.split=test generation.profiles=[tdla,tdld,cdla] generation.include_mixed=false generation.num_samples=32768 generation.batch_size=512 generation.base_seed=67 generation.output_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
