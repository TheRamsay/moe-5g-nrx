#!/usr/bin/env bash
#PBS -N latency-all-real
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb
#PBS -l walltime=04:00:00
#PBS -j oe

# Wall-clock latency benchmark — all 4 models, synthetic + real test data (uma + tdlc).
# All models run on the SAME node so speedup ratios are valid.
# 4h walltime: 4 models * ~45min each (PTX JIT cold per model).

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="scripts/benchmark_latency.py"
export RUN_ARGS="--device cuda --batch-size 64 --n-iter 200 --n-warmup 20 --data-dir /storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test --out latency_all_real.json"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
