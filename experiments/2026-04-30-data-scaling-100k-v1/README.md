# Data Scaling — exp26 Recipe at 100k Samples (v1)

## Question

Does training on 100k samples (2× current 50k) meaningfully improve the
exp26 result, or has BLER plateaued?

## Background

Teacher raised concern that 50k training samples might be too small. Most NRX
papers use 10k–100k, so 50k is reasonable but defensible. Empirical answer is
the cleanest defense: re-run the exp26 recipe at 2× scale and measure.

The full HuggingFace dataset (`Vack0/moe-5g-nrx`) has ~125k samples per profile
(~250k total mixed). 50k is our 20% subset. 100k is achievable with 64 GB RAM
nodes; full 250k would need streaming or larger memory.

## Configs

Two-step pipeline:

1. `submit_export.sh` — one-time export of 100k samples per profile from the
   cached HF Arrow files into `train-100k-array3d/` directory. ~30-60 min.
2. `submit_train.sh` — exp40 training: same recipe as exp26 (α=2e-3, asym
   warm-start, seed 67) but `hf_max_samples=100000` instead of 50000.

## Cluster

- Export: `select=1:ncpus=4:mem=96gb:scratch_ssd=120gb`, walltime 4h
- Train: `select=1:ncpus=8:ngpus=1:mem=96gb:scratch_ssd=80gb`, walltime 5h
  (96 GB RAM to comfortably hold the 42 GB Arrow table + worker copies)

## Expected outcomes

- **100k matches 50k** → 50k was sufficient, defend the result. Strong story
  for the consultation: "we tested at 2x scale, no improvement, 50k is enough".
- **100k significantly better** → re-run all final headline results at 100k.

## Status

Export job 19583194 (in flight at time of writing). Train job to be submitted
once export finishes.
