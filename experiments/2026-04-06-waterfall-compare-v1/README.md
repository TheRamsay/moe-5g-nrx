# Waterfall BLER Comparison (2026-04-06)

## Question

Does expert size produce a meaningful BLER gap in the waterfall region (the narrow SNR band
where BLER transitions from ~1.0 to ~0), even though average BER/BLER look nearly identical?

## Why This Matters

The MoE approach only works if different expert sizes produce different quality outputs at
specific SNR operating points. Prior experiments showed nano vs large have nearly identical
average BER (~0.7pp gap) and modest average BLER (~3pp gap). But the 5-bin SNR resolution
was too coarse to see the waterfall clearly — the entire transition lands in one bin.

Preliminary data from the highest SNR bin (bin 4) shows:
- TDLC SNR=17: nano BLER=0.674, large_s32 BLER=0.509 (**16.5pp gap**)
- UMA SNR=22: nano BLER=0.845, large_s32 BLER=0.823 (2.2pp gap)

This experiment uses 2 dB SNR bins to resolve the full waterfall curve.

## Models Compared

| Label | Params | state_dim | Artifact |
|---|---:|---:|---|
| nano | 90k | 32 | `model-dense_nano_final20k_...-aos4hhid:best` |
| small | 168k | 56 | `model-dense_small_final20k_...-kivdz4qu:best` |
| large_s32 | 363k | 32 | `model-dense_large_s32_final20k_...-rdfefyt1:best` |
| large_s56 | 450k | 56 | `model-dense_large_final20k_...-55l1dpby:best` |

## Setup

- Data: `dense-v1/val/{tdlc,uma}.pt` (same cached val data as all other experiments)
- SNR bins: every 2 dB across full range (TDLC: -10 to 20, UMA: -5 to 25)
- Output: `waterfall.png` + console table

## Findings

*Pending — fill in after job completes.*
