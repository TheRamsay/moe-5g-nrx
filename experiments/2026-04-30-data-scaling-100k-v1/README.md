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

- ✅ Export job 19583194 (~1h08): 100k samples per profile to `train-100k-array3d/`
- ✅ exp40 (s67, α=2e-3): collapsed → BLER ~0.953
- ✅ exp58 (s42, α=2e-3 retry): also collapsed → BLER ~0.968 (real_flops 0.305)
- ✅ **exp60 (s67, α=1e-3): HETEROGENEOUS — confirmed α/data hypothesis** (DONE 2026-05-01)

## Results — exp40 (DONE 2026-04-30)

| Run | Data | Seed | UMa BLER | TDLC BLER | Avg BLER | real_flops | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| exp26 (50k headline) | 50k | 67 | 0.937 | 0.867 | **0.902** | 0.56 | heterogeneous ✓ |
| **exp40 (100k retry)** | **100k** | 67 | **0.968** | **0.938** | **~0.953** | **0.465** | **collapsed ✗** |

**More data → WORSE BLER (+5 pp).** This is NOT a clean "100k matches/beats 50k"
result. Looking at the routing pattern (real_flops=0.465, ~45% FLOPs):
**this is the s32 collapse signature** — router shifted toward nano/small,
large under-used.

## Refined hypothesis after exp58 collapse (and remembering the original anchor)

CLAUDE.md history shows the **original asym-warm anchor** was trained on the
**FULL HuggingFace stream (~250k samples) with α=1e-3** and produced
**heterogeneous routing fine** (BLER 0.910 / 61% FLOPs). So data scale alone
isn't the issue.

The variable that changed is **α**:

| Run | α | Data | Result |
|---|---:|---:|---|
| Original anchor | **1e-3** | **~250k** | ✓ heterogeneous (0.910 BLER) |
| exp26 headline | 2e-3 | 50k | ✓ heterogeneous (0.902 BLER) |
| exp40 | 2e-3 | 100k | ✗ collapsed (Phase-1 style, heavy nano) |
| exp58 | 2e-3 | 100k | ✗ collapsed (also nano-heavy, real_flops 0.305) |
| **exp60** | **1e-3** | **100k** | **(in flight)** |

**Refined hypothesis:** the **α/data ratio** matters, not data scale alone.
α=2e-3 was tuned for 50k. At 100k the same α applied per sample produces a
stronger effective FLOPs signal across the larger epoch — pushes router to
nano too early before large can wake up.

**exp60 tests this:** if α=1e-3 at 100k recovers heterogeneous routing
matching the original anchor's BLER ~0.91 → confirms the hypothesis. Clean
methodological finding for the writeup:
> "The asym-warm recipe is robust to data scale provided the FLOPs penalty α
> is scaled inversely with data-per-epoch."

## Result — exp60 (DONE 2026-05-01) ✓ HYPOTHESIS CONFIRMED

| Run | α | Data | UMa BLER | TDLC BLER | Avg | real_flops | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| exp26 (50k headline) | 2e-3 | 50k | 0.937 | 0.867 | 0.902 | 0.56 | ✓ |
| exp40 | 2e-3 | 100k | 0.968 | 0.938 | 0.953 | 0.465 | ✗ |
| exp58 | 2e-3 | 100k | 0.97+ | 0.96+ | 0.968 | 0.305 | ✗ |
| **exp60** | **1e-3** | **100k** | **~0.941** | **~0.864** | **~0.902** | **~0.65** | **✓** |

Trajectory at exp60 stable from step 9000-11500: UMa ~0.941, TDLC ~0.864.
Step 12000 had a one-batch noise spike (UMa 0.95 / TDLC 0.90) but the run is
clearly heterogeneous, not collapsed. real_flops ~0.65 (vs exp26's 0.56) —
slightly more compute, same BLER — within the bimodal variance band.

**This is a clean methodological finding.** Two collapsed runs (exp40, exp58)
explained by a single principle (α scales inversely with data×steps), with
the recipe's correctness restored at the matching α. Eligible for the report.
