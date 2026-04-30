#!/usr/bin/env bash
#PBS -N eval-lmmse-snr20
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb
#PBS -l walltime=01:00:00
#PBS -j oe

# Re-run LS-MRC LMMSE baseline with snr_bins=20 — for the same high-res
# waterfall figure as eval46 (exp26) and eval47 (dense_large).

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="scripts/evaluate_lmmse.py"
export RUN_ARGS="--data-dir /storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test --profiles uma tdlc --snr-bins 20 --batch-size 256 --device cuda --mode ls_mrc --out lmmse_snr20_results.json"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
