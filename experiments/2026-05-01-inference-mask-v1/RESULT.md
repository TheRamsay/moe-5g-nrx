# Inference-Mask RESULT — Training-scaffold hypothesis CONFIRMED ✓

## Verified facts (`docs/figures/inference_mask_results.json`, 32k samples per profile)

Loaded exp26 best ckpt (`19457671/checkpoints/compute_aware_moe_nrx_best.pt`)
and ran 3 inference modes on the locked dense-v1 test set.

| Mode | UMa BLER | TDLC BLER | **Avg BLER** | Avg FLOPs ratio | Routing change |
|---|---:|---:|---:|---:|---|
| **baseline** (exp26 unchanged) | 0.93692 | 0.86734 | **0.90213** | **0.5578** | — |
| **A_mask** (force {nano, large}) | 0.93692 | 0.86722 | **0.90207** | 0.5521 | small→0%; redistributed |
| **B_sink** (zero-out small's slot) | 0.93692 | 0.86737 | **0.90215** | **0.4732** | small kept routed but produces zeros |

**BLER differences across modes:** **0.00008 (essentially zero — within sample noise).**

**Mode B yields a clean −9pp FLOPs reduction at the same BLER.**

## Detailed per-profile metrics

### UMa (32,768 samples)

| Mode | BLER | BER | FLOPs ratio | Routing (n/s/l) |
|---|---:|---:|---:|---|
| baseline | 0.93692 | 0.27456 | 0.4659 | 48% / 26% / 26% |
| A_mask | 0.93692 | 0.27575 | 0.4502 | 69% / 0% / 31% |
| B_sink | 0.93692 | 0.33458 | **0.3998** | 48% / 26%→sink / 26% |

### TDLC (32,768 samples)

| Mode | BLER | BER | FLOPs ratio | Routing (n/s/l) |
|---|---:|---:|---:|---|
| baseline | 0.86734 | 0.12643 | 0.6498 | 15% / 40% / 44% |
| A_mask | 0.86722 | 0.12845 | 0.6540 | 43% / 0% / 57% |
| B_sink | 0.86737 | 0.24144 | **0.5465** | 15% / 40%→sink / 44% |

## Why BLER is identical across modes

Per the middle-expert investigation (2026-05-01):
- Small's routed samples have BLER 1.0 (block-level fail) regardless of which expert handles them
- Even forced to large, those samples STILL fail at block level (counterfactual confirmed)
- So replacing small's bits with random/zero outputs doesn't change BLER — those blocks fail either way

Mode B's higher BER (0.288 avg vs 0.200 baseline) reflects that random/zero
predictions make ~50% errors per bit, but this only affects the BIT-error
metric, not the block-decode success. Since BLER is what matters for actual
transmission outcomes (HARQ retransmission triggers on block fail), BER
deterioration on already-failing blocks is irrelevant.

## Why Mode A reduces FLOPs less than Mode B

Mode A redirects small's traffic to {nano, large} based on the router's
secondary preference. On TDLC, 57% of small's traffic goes to large
(EXPENSIVE — 1604M vs small's 695M), partially offsetting the savings
from the 43% going to nano (cheap — 320M). Net TDLC FLOPs actually goes
UP slightly (65.4% vs 65.0% baseline) before being averaged with UMa savings.

Mode B keeps the router's original choice but charges 0 FLOPs for
small-routed samples (sink). Cleaner — saves the entire 695M cost of
small whenever the router picks it.

## Headline result for the report

> **exp26 + inference Mode B is a strict Pareto improvement over exp26
> baseline:** identical avg BLER (0.9021), 9pp lower avg FLOPs (0.4732 vs
> 0.5578). No retraining required — just modify the inference loop to
> replace small's expert with a zero-output operator.

This is a **training-scaffold** finding: the 3-decoder MoE is necessary at
TRAINING time for routing-policy learning (gradient signal from small's
partial decodes), but at INFERENCE time the small expert's output is
discardable because its routed samples fail at block level anyway.

## Cross-validation with sink-in-training experiments

Three attempts to incorporate sink into the architecture during training
all failed:

| Attempt | Train | Outcome |
|---|---|---|
| exp61 v2 | sink + channel_only + large | FLOPs +22pp WORSE (router defaults to large) |
| exp64 | sink + small + large | TOTAL collapse to all-sink |
| exp65 | sink + nano + large | Training unstable, BLER +7pp |

**Sink belongs at inference, not at training.** Verified across three
independent architecture attempts.
