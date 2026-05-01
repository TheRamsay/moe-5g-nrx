#!/usr/bin/env bash
#PBS -N inference-mask-pareto
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb
#PBS -l walltime=01:30:00
#PBS -j oe

# Apply Mode A + B inference modifications across multiple alpha-sweep checkpoints.
# Tests whether the training-scaffold principle (B yields free FLOPs reduction)
# generalizes across alpha values — exp25 (a=1e-3), exp26 (a=2e-3), exp27 (a=5e-3).
# Also outputs per-SNR breakdown for the headline figure.

cd /storage/brno2/home/ramsay/moe-5g-nrx
PY=/storage/brno2/home/ramsay/.local/bin/uv
DATA=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test
OUT=/storage/brno2/home/ramsay/moe-5g-nrx/docs/figures

# exp25 (alpha=1e-3)
$PY run --offline --python 3.10 python scripts/evaluate_inference_mask.py \
  --checkpoint /storage/brno2/home/ramsay/moe-5g-nrx/results/19457670.pbs-m1.metacentrum.cz/checkpoints/compute_aware_moe_nrx_best.pt \
  --data-dir $DATA \
  --out-json $OUT/inference_mask_exp25.json \
  --max-samples 32768 --device cuda

# exp26 (alpha=2e-3) — re-run with per-SNR
$PY run --offline --python 3.10 python scripts/evaluate_inference_mask.py \
  --checkpoint /storage/brno2/home/ramsay/moe-5g-nrx/results/19457671.pbs-m1.metacentrum.cz/checkpoints/compute_aware_moe_nrx_best.pt \
  --data-dir $DATA \
  --out-json $OUT/inference_mask_exp26.json \
  --max-samples 32768 --device cuda

# exp27 (alpha=5e-3)
$PY run --offline --python 3.10 python scripts/evaluate_inference_mask.py \
  --checkpoint /storage/brno2/home/ramsay/moe-5g-nrx/results/19457672.pbs-m1.metacentrum.cz/checkpoints/compute_aware_moe_nrx_best.pt \
  --data-dir $DATA \
  --out-json $OUT/inference_mask_exp27.json \
  --max-samples 32768 --device cuda
