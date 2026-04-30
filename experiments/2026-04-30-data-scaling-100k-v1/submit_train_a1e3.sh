#!/usr/bin/env bash
#PBS -N moe-100k-a1e3
#PBS -l select=1:ncpus=8:ngpus=1:mem=96gb:scratch_ssd=80gb
#PBS -l walltime=05:00:00
#PBS -j oe

# exp60: 100k samples × α=1e-3 (matches original asym-warm anchor's α).
# Tests whether the data-scale collapse of exp40/58 was due to α=2e-3 being
# over-tuned for 100k, not data scale per se.

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="main.py"
export RUN_ARGS="experiment=exp60_moe_a1e3_100k_s67 runtime.device=cuda validation.data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/val training.hf_train_data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/train-100k-array3d training.hf_max_samples=100000"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
