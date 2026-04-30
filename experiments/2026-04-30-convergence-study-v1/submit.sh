#!/usr/bin/env bash
#PBS -N exp59-30k-convergence
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb
#PBS -l walltime=06:00:00
#PBS -j oe

# exp59: 30k convergence run. Same recipe as exp26 (alpha=2e-3, asym warm,
# seed 67) but 2.5× the training steps. Tests whether 12k is converged and
# produces the final-report headline number.

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="main.py"
export RUN_ARGS="experiment=exp59_moe_a2e3_30k_s67 runtime.device=cuda validation.data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/val training.hf_train_data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/train-50k-array3d training.hf_max_samples=50000"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
