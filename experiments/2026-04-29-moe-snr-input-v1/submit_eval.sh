#!/usr/bin/env bash
#PBS -N eval-exp38-snr
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb
#PBS -l walltime=01:00:00
#PBS -j oe

# Test-set eval for exp38 (SNR-proxy input statistics in router).
# Checkpoint from training job 19548638.

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="scripts/evaluate.py"
export RUN_ARGS="experiment=eval45_exp38_snr_input evaluation.checkpoint=/storage/brno2/home/ramsay/moe-5g-nrx/results/19548638.pbs-m1.metacentrum.cz/checkpoints/compute_aware_moe_nrx_best.pt evaluation.checkpoint_artifact=null evaluation.data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test validation.data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/val runtime.device=cuda"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
