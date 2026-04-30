#!/usr/bin/env bash
#PBS -N moe-100k-train
#PBS -l select=1:ncpus=8:ngpus=1:mem=96gb:scratch_ssd=80gb
#PBS -l walltime=05:00:00
#PBS -j oe

# exp40: train exp26 recipe (alpha=2e-3, asym warm, seed 67) at 100k samples.
# Tests the data-scaling concern. Wait for the export job to finish first.
# Memory: 96 GB to comfortably hold the 100k Arrow table (~42 GB).
# Walltime: 5h to be safe (12k steps at ~2x current data ~= 2-3h).

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="main.py"
export RUN_ARGS="experiment=exp40_moe_a2e3_100k_s67 runtime.device=cuda validation.data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/val training.hf_train_data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/train-100k-array3d training.hf_max_samples=100000"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
