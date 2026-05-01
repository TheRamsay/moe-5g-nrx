#!/usr/bin/env bash
#PBS -N exp65-sink-nano-large
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb
#PBS -l walltime=03:00:00
#PBS -j oe

# exp65: bold simplification {sink, nano, large}.
# Tests user's hypothesis: nano can serve both hopeless + borderline regimes
# (middle-expert investigation showed nano BER=0.237 ≈ small BER=0.230 on
# small's routed samples). Drop small entirely.

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="main.py"
export RUN_ARGS="experiment=exp65_sink_nano_large runtime.device=cuda validation.data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/val training.hf_train_data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/train-50k-array3d training.hf_max_samples=50000"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
