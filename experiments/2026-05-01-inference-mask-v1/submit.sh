#!/usr/bin/env bash
#PBS -N inference-mask-AB
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb
#PBS -l walltime=01:00:00
#PBS -j oe

# Inference-time routing modifications on exp26 checkpoint.
# Mode A: mask small at inference, force {nano, large}.
# Mode B: replace small's expert output with zeros (sink-style).

cd /storage/brno2/home/ramsay/moe-5g-nrx
/storage/brno2/home/ramsay/.local/bin/uv run --offline --python 3.10 python scripts/evaluate_inference_mask.py \
  --checkpoint knn_moe-5g-nrx/moe-5g-nrx/model-moe_alphasweep_asym_a2e3_s67-t6lkdep2:best \
  --data-dir /storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test \
  --out-json /storage/brno2/home/ramsay/moe-5g-nrx/docs/figures/inference_mask_results.json \
  --max-samples 32768 \
  --device cuda
