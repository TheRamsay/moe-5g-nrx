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
  --checkpoint /storage/brno2/home/ramsay/moe-5g-nrx/results/19457671.pbs-m1.metacentrum.cz/checkpoints/compute_aware_moe_nrx_best.pt \
  --data-dir /storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test \
  --out-json /storage/brno2/home/ramsay/moe-5g-nrx/docs/figures/inference_mask_results.json \
  --max-samples 32768 \
  --device cuda
