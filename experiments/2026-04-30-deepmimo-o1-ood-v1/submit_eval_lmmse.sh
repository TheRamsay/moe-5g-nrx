#!/usr/bin/env bash
#PBS -N eval-lmmse-o1-ood
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb
#PBS -l walltime=01:00:00
#PBS -j oe

# LMMSE LS-MRC baseline on DeepMIMO O1_3p5 OOD test set.

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="scripts/evaluate_lmmse.py"
export RUN_ARGS="--data-dir /storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test --profiles o1_3p5 --snr-bins 7 --batch-size 256 --device cuda --mode ls_mrc --out lmmse_o1_ood_results.json"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
