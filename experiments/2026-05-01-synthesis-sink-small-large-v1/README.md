# Synthesis Architecture — `{sink, small, large}` (exp64)

## Why

Two prior findings combined:

1. **Per-expert success rate (2026-04-30):** nano and small NEVER decode (0% success
   on routed samples). Only large decodes (23-29%). Suggested nano/small are pure
   compute optimizers replaceable by lighter alternatives.

2. **exp61 v2 (2026-05-01):** Replaced nano + small with sink + channel_only.
   BLER matched exp26 (0.900 vs 0.902) but **FLOPs collapsed to 0.78 (vs exp26's 0.56)**.
   The router defaulted to large for any non-hopeless sample because channel_only
   could not provide gradient signal as a "middle compute tier" decode attempt.

**Synthesis:** the router needs DECODE attempts at multiple compute tiers (gradient
signal), not just any cheap fallback. Replace nano (320M total FLOPs, 0% decode) with
sink (true zero-cost skip), but keep small as the productive middle tier.

## Architecture

| Expert | Params | Class | Total FLOPs |
|---|---:|---|---:|
| sink | 0 | _SinkExpert | 0 |
| small | 115k | _ExpertHead (full decoder) | 695M (stem 285M + expert 410M) |
| large | 370k | _ExpertHead (full decoder) | 1604M (stem 285M + expert 1319M) |
| **TOTAL** | **572k** | | |

Versus exp26 reference: nano (90k, 320M) + small (115k, 695M) + large (370k, 1604M) = 583k.

## Predicted outcomes

- **Best case:** BLER ≈ exp26 (~0.902) at FLOPs ≤ exp26 (~0.50-0.56). Sink replaces
  nano with zero compute, saving 35M per nano-route. Modest but real FLOPs win.
  Cleaner final architecture for the report.
- **BLER same / FLOPs same:** sink-vs-nano substitution is neutral; the architectural
  story stays "compute tiers matter" but exp26 remains headline.
- **BLER worse:** indicates nano's specific 320M tier was load-bearing — small
  alone can't fill the gap. Confirms 3-decoder design is irreducible.

## Recipe

Same as exp26: α=2e-3, asym warm-start (warm small from dense_small,
cold large), seed 67, 12k steps. Only architectural change: nano → sink.

## Cluster

`select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb`, walltime 3h.
