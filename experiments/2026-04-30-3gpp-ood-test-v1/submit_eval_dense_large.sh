#!/usr/bin/env bash
#PBS -N eval-dense-large-3gpp-ood
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb
#PBS -l walltime=01:00:00
#PBS -j oe

# eval53: dense_large on TDL-A + TDL-D + CDL-A (in-family OOD).

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="scripts/evaluate.py"
export RUN_ARGS="experiment=eval53_3gpp_ood_dense_large evaluation.checkpoint_artifact=knn_moe-5g-nrx/moe-5g-nrx/model-dense_large_final20k_constant_lr_s67-55l1dpby:best evaluation.data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test validation.data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/val runtime.device=cuda"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
