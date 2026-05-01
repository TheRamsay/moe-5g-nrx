#!/usr/bin/env bash
#PBS -N eval-exp60-test
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb
#PBS -l walltime=01:00:00
#PBS -j oe

# eval62: test-set eval of exp60 (100k + alpha=1e-3) — locks headline number.

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="scripts/evaluate.py"
export RUN_ARGS="experiment=eval62_test_exp60 evaluation.checkpoint=/storage/brno2/home/ramsay/moe-5g-nrx/results/19587197.pbs-m1.metacentrum.cz/checkpoints/compute_aware_moe_nrx_best.pt evaluation.data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test validation.data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/val runtime.device=cuda"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
