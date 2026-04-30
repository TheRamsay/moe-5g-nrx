#!/usr/bin/env bash
#PBS -N gen-deepmimo-o1
#PBS -l select=1:ncpus=8:ngpus=0:mem=32gb:scratch_ssd=20gb
#PBS -l walltime=03:00:00
#PBS -j oe

# Generate DeepMIMO O1_3p5 OOD test data.
# O1 = outdoor street canyon at 3.5 GHz (same carrier as our training).
# Simpler geometry than ASU campus — diagnostic test for whether ASU was
# the specifically-hard scenario or all ray-traced outdoor fails.
#
# Output: /storage/.../moe-5g-datasets/dense-v1/test/o1_3p5.pt (32k samples)
# Deterministic seed (base_seed=67 + TEST_SEED_OFFSET) matches existing
# in-distribution test data.

export REPO_ROOT="${PBS_O_WORKDIR}"
export ENTRYPOINT="scripts/generate_deepmimo_dataset.py"
export RUN_ARGS="generation.split=test generation.num_samples=32768 generation.batch_size=512 generation.base_seed=67 generation.profiles=[o1_3p5] generation.include_mixed=false generation.output_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1 generation.deepmimo.scenario=O1_3p5 generation.deepmimo.dataset_folder=/storage/brno2/home/ramsay/moe-5g-datasets/deepmimo-train"

bash "${PBS_O_WORKDIR}/scripts/metacentrum_job.sh"
