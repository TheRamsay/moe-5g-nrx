# High-Resolution Per-SNR Re-evaluation (v1)

## Question

Does the BLER@SNR comparison between exp26, dense_large, and LS-MRC look
clean at finer SNR resolution (1.5 dB per bin instead of 5 dB)?

## Background

Default eval uses `snr_bins=7` per profile, which gives ~5 dB per bin —
too coarse to characterise the waterfall transition (5–18 dB on TDLC).
Re-running the same eval at `snr_bins=20` gives ~1.5 dB per bin: enough
resolution to draw a smooth waterfall curve.

This is one of the three concrete future-work items identified
post-consultation. Cheap to do (no new training, just re-eval at higher
resolution) and high-value (cleaner waterfall figure for the report).

## Configs

| Submit | Model | Resolution |
|---|---|---:|
| `submit_exp26.sh` (eval46) | exp26 MoE checkpoint | 20 SNR bins |
| `submit_dense_large.sh` (eval47) | dense_large baseline | 20 SNR bins |
| `submit_lmmse.sh` | LS-MRC classical (no NN) | 20 SNR bins |

All three use the existing test set (32k samples per profile, UMa+TDLC),
so the per-bin sample count is reduced from ~4.6k → ~1.6k per bin — still
enough for stable BLER statistics.

## Cluster

- Resources: `select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb`
- Walltime: 1h each (eval is fast — ~5-10 min once allocated)

## Results

Aggregate BLER (matches earlier 7-bin runs as expected):

| Model | UMa BLER | TDLC BLER | Avg BLER |
|---|---:|---:|---:|
| exp26 | 0.937 | 0.867 | 0.902 |
| dense_large | 0.936 | 0.866 | 0.901 |
| LMMSE (LS-MRC) | 0.939 | 0.861 | 0.900 |

Per-SNR breakdowns (20 bins each) are in W&B for plotting. LMMSE per-SNR
example shows clean drops:
- TDLC SNR 11-13 dB: BLER ~0.998
- TDLC SNR 16-18 dB: BLER ~0.155
- TDLC SNR 18-20 dB: BLER ~0.027

These breakdowns enable a proper waterfall figure for the consultation.
