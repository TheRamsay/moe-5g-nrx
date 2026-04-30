#!/usr/bin/env bash
#PBS -N lmmse-baseline
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb
#PBS -l walltime=01:00:00
#PBS -j oe

# Classical LS-MRC + max-log-LLR baseline evaluation on uma + tdlc test sets.
# Standalone — no checkpoint loading, no W&B init. Just runs the algorithm
# and dumps a JSON. ~5 min runtime expected.

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="scripts/evaluate_lmmse.py"
export RUN_ARGS="--data-dir /storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test --profiles uma tdlc --snr-bins 7 --batch-size 256 --device cuda --out lmmse_eval_results.json"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
