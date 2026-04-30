#!/usr/bin/env bash
#PBS -N moe-snr-input
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb
#PBS -l walltime=03:00:00
#PBS -j oe

# exp38: MoE with explicit SNR-proxy input statistics in router.
# Identical to exp26 (Pareto knee, α=2e-3, asym warm-start, seed 67)
# except router also receives channel_power + channel_variance scalars.
# 3h walltime; 12k steps ≈ 2h on GPU.

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="main.py"
export RUN_ARGS="experiment=exp38_moe_snr_input_a2e3_s67 runtime.device=cuda validation.data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/val training.hf_train_data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/train-50k-array3d training.hf_max_samples=50000"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
