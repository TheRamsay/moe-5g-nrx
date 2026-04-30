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

- ✅ Export job 19583194 finished 2026-04-30 (~1h08): saved 100k samples per
  profile to `/storage/.../moe-5g-datasets/train-100k-array3d/{uma,tdlc}`
  (21.5 GB each).
- ✅ Train job 19585019 (exp40, seed 67) finished 2026-04-30.
- 🔄 Train job 19586443 (exp58, seed 42 retry) submitted to test bimodality.

## Results — exp40 (DONE 2026-04-30)

| Run | Data | Seed | UMa BLER | TDLC BLER | Avg BLER | real_flops | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| exp26 (50k headline) | 50k | 67 | 0.937 | 0.867 | **0.902** | 0.56 | heterogeneous ✓ |
| **exp40 (100k retry)** | **100k** | 67 | **0.968** | **0.938** | **~0.953** | **0.465** | **collapsed ✗** |

**More data → WORSE BLER (+5 pp).** This is NOT a clean "100k matches/beats 50k"
result. Looking at the routing pattern (real_flops=0.465, ~45% FLOPs):
**this is the s32 collapse signature** — router shifted toward nano/small,
large under-used.

## Two interpretations

1. **Bad luck:** The asym-warm bimodality strikes again. Same recipe, different
   data scale, hit the bad attractor. Earlier 3-seed analysis showed 1/3 seeds
   collapse — exp40 likely hit that 1/3 outcome.
2. **Data scale exacerbates instability:** More data could shift the loss
   landscape in ways that make the bad attractor easier to reach.

## Why we queued exp58 (s42 retry)

To distinguish between the two interpretations:
- If exp58 (seed 42 at 100k) succeeds → bimodality, more data is fine on
  average, exp40 was unlucky.
- If exp58 also collapses → data scale really does worsen instability.

Either result is informative for the writeup.

## Honest framing

> "100k training also exhibited the asym-warm bimodality — landed in the
> bad attractor. Without multiple seeds at 100k we can't distinguish 'bad
> luck' from 'more data hurts the recipe.' To make a clean data-scaling
> claim we'd need ≥3 seeds at 100k. Honest answer: 50k vs 100k results
> are inconclusive due to recipe instability, not a clean answer."
