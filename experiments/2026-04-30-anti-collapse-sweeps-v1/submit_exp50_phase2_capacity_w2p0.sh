#!/usr/bin/env bash
#PBS -N phase2-capacity-w2p0
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb
#PBS -l walltime=03:00:00
#PBS -j oe

# Anti-collapse sweep: exp50_phase2_capacity_w2p0.
# Phase 2 v1 base + the sweep parameter; tests whether properly-tuned
# anti-collapse mechanism prevents the lock-in to large.

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="main.py"
export RUN_ARGS="experiment=exp50_phase2_capacity_w2p0 runtime.device=cuda validation.data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/val training.hf_train_data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/train-50k-array3d training.hf_max_samples=50000"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
