#!/usr/bin/env bash
#PBS -N exp56-asym-coldsmall
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb
#PBS -l walltime=03:00:00
#PBS -j oe

# exp56: asym-warm with cold-SMALL (warm nano + warm large).
# Symmetric counterpart to exp26 (cold-large). Tests the generalization
# of the asym-warm principle.

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="main.py"
export RUN_ARGS="experiment=exp56_asym_coldsmall_a2e3_s67 runtime.device=cuda validation.data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/val training.hf_train_data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/train-50k-array3d training.hf_max_samples=50000"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
