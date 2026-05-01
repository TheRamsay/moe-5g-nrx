#!/usr/bin/env bash
#PBS -N middle-expert-investigation
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb
#PBS -l walltime=01:00:00
#PBS -j oe

# Middle-expert investigation: counterfactual per-expert outputs on exp26 test set.
# Answers: what does small ACTUALLY produce when routed? Bits, channel, confidence?

cd /storage/brno2/home/ramsay/moe-5g-nrx
/storage/brno2/home/ramsay/.local/bin/uv run --offline --python 3.10 python scripts/analyze_middle_expert.py \
  --checkpoint knn_moe-5g-nrx/moe-5g-nrx/model-moe_alphasweep_asym_a2e3_s67-t6lkdep2:best \
  --data-dir /storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test \
  --out-dir /storage/brno2/home/ramsay/moe-5g-nrx/docs/figures \
  --max-samples 4000 \
  --device cuda
