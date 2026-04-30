#!/usr/bin/env bash
#PBS -N genie-mrc-baseline
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb
#PBS -l walltime=01:00:00
#PBS -j oe

# Genie-MRC: same algorithm as LS-MRC but uses TRUE channel from Sionna
# (channel_target). Upper bound for classical receivers.

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="scripts/evaluate_lmmse.py"
export RUN_ARGS="--data-dir /storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test --profiles uma tdlc --snr-bins 7 --batch-size 256 --device cuda --mode genie_mrc --out genie_mrc_eval_results.json"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
