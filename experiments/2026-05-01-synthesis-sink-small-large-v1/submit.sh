#!/usr/bin/env bash
#PBS -N exp64-sink-small-large
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb
#PBS -l walltime=03:00:00
#PBS -j oe

# exp64: synthesis architecture {sink, small, large}.
# Replaces exp26's nano with sink (true zero-cost skip), keeps small as
# productive middle compute tier (channel_only failed that role in exp61 v2).

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="main.py"
export RUN_ARGS="experiment=exp64_sink_small_large runtime.device=cuda validation.data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/val training.hf_train_data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/train-50k-array3d training.hf_max_samples=50000"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
