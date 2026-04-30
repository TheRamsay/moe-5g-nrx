#!/usr/bin/env bash
#PBS -N single-ant-baseline
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb
#PBS -l walltime=01:00:00
#PBS -j oe

# Single-antenna detection: use only first receive antenna, no MRC diversity.
# Naive lower-bound classical baseline.

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="scripts/evaluate_lmmse.py"
export RUN_ARGS="--data-dir /storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test --profiles uma tdlc --snr-bins 7 --batch-size 256 --device cuda --mode single_ant --out single_ant_eval_results.json"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
