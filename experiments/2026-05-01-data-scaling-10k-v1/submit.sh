#!/usr/bin/env bash
#PBS -N exp63-10k-a2e3
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb
#PBS -l walltime=03:00:00
#PBS -j oe

# exp63: 10k samples × α=2e-3 — lower-bound data-scaling test.
# Reuses train-50k-array3d cache, just truncates with hf_max_samples=10000.

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="main.py"
export RUN_ARGS="experiment=exp63_moe_a2e3_10k_s67 runtime.device=cuda validation.data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/val training.hf_train_data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/train-50k-array3d training.hf_max_samples=10000"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
