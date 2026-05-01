# Inference-Mask Across Alpha Sweep + Per-SNR

## Two questions in one job

**Q1 (per-SNR):** Does the BLER preservation under Mode B hold across all SNR
bins, or only on average? If small matters at extreme SNR, Mode B would
hurt that bin specifically.

**Q2 (cross-α):** Does the "training scaffold" principle generalize across
the alpha sweep? If Mode B gives free FLOPs reduction on exp25 and exp27 too,
the finding is robust. If only on exp26, it's specific to that recipe.

## Method

Runs `scripts/evaluate_inference_mask.py` (already extended with per-SNR
binning) on three checkpoints:

| Exp | α | Checkpoint job | Modes |
|---|---:|---:|---|
| exp25 | 1e-3 | 19457670 | baseline / A_mask / B_sink |
| exp26 | 2e-3 | 19457671 | (re-run for per-SNR) |
| exp27 | 5e-3 | 19457672 | baseline / A_mask / B_sink |

Each produces a separate JSON output with overall + per-SNR-bin metrics.

## Outputs

- `docs/figures/inference_mask_exp25.json`
- `docs/figures/inference_mask_exp26.json` (overwrites earlier exp26 run, now with per-SNR)
- `docs/figures/inference_mask_exp27.json`

## Cluster

`select=1:ncpus=4:ngpus=1:mem=16gb`, walltime 1.5h. ~5 min per checkpoint.

## What success looks like

**Best case:** Mode B gives ~9pp FLOPs reduction on all three checkpoints,
BLER preserved within 0.001 in every SNR bin. → "training scaffold" is a
robust principle, not a one-off.

**Specific case:** Mode B works on exp26 but fails on exp25/exp27. → Effect
is recipe-specific; reframe the contribution.

**Per-SNR case:** Mode B preserves BLER on average but fails at one extreme
SNR. → Mode B has a hidden tradeoff; needs caveat.
