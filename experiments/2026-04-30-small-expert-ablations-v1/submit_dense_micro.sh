#!/usr/bin/env bash
#PBS -N dense-micro-pretrain
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb
#PBS -l walltime=04:00:00
#PBS -j oe

# exp42: pretrain dense_micro (block_dim=16, 8 blocks) for 20k steps.
# Used as warm-start checkpoint for exp43 (smaller-small MoE ablation).

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="main.py"
export RUN_ARGS="experiment=exp42_dense_micro_final20k runtime.device=cuda validation.data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/val training.hf_train_data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/train-50k-array3d training.hf_max_samples=50000"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
