#!/usr/bin/env bash
#PBS -N eval-exp26-3gpp-ood
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb
#PBS -l walltime=01:00:00
#PBS -j oe

# eval52: exp26 on TDL-A + TDL-D + CDL-A (in-family OOD).

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="scripts/evaluate.py"
export RUN_ARGS="experiment=eval52_3gpp_ood_exp26 evaluation.checkpoint_artifact=knn_moe-5g-nrx/moe-5g-nrx/model-moe_alphasweep_asym_a2e3_s67-t6lkdep2:best evaluation.data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test validation.data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/val runtime.device=cuda"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
