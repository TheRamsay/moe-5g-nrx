#!/usr/bin/env bash
#PBS -N o1-arrow-to-pt
#PBS -l select=1:ncpus=4:mem=64gb:scratch_ssd=20gb
#PBS -l walltime=01:00:00
#PBS -j oe

# Convert the O1_3p5 Arrow dataset to .pt format expected by evaluate.py.

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="scripts/arrow_to_pt.py"
export RUN_ARGS="--arrow-dir /storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test/O1_3p5 --out /storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test/o1_3p5.pt --profile o1_3p5"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
