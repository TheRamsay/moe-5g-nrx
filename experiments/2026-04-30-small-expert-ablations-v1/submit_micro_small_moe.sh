#!/usr/bin/env bash
#PBS -N moe-micro-small
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb
#PBS -l walltime=03:00:00
#PBS -j oe

# exp43: MoE with {nano, MICRO-small (block_dim=16), large}.
# REQUIRES: exp42 (dense_micro pretrain) finished and uploaded as W&B artifact.
# Same exp26 recipe otherwise.

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="main.py"
export RUN_ARGS="experiment=exp43_moe_micro_small_a2e3 runtime.device=cuda validation.data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/val training.hf_train_data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/train-50k-array3d training.hf_max_samples=50000"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
