# Inference-Time Routing Modifications

## Why

Middle-expert investigation (2026-05-01) showed that on small's routed samples:
- nano BER 0.237 ≈ small BER 0.230 (essentially identical)
- ALL three experts fail at block level (BLER 1.0 universal)

So at INFERENCE, replacing small's traffic with nano's might preserve BLER
while saving FLOPs. This script tests that hypothesis WITHOUT retraining,
using the exp26 checkpoint and modifying only the routing decision at eval time.

## Two modes

| Mode | What changes at inference |
|---|---|
| **A_mask** | Set router's small-logit to -∞ → argmax forces choice between {nano, large} |
| **B_sink** | Keep router's natural choice but if it picks small, output zeros & charge 0 expert FLOPs |

Both modes test the "training scaffold" hypothesis: small was useful during
training (gradient signal) but may be discardable at inference.

## Predicted outcomes

**Mode A (mask):**
- Some samples re-route to nano (cheaper) → FLOPs ↓
- Some samples re-route to large (more expensive) → FLOPs ↑
- Net FLOPs depends on router's preference between nano vs large for small-natural samples
- BLER probably +0.5 to +2pp vs exp26 (some borderline samples now route worse)

**Mode B (sink):**
- All small-routed samples get zero output → BLER definitely worse on those
- FLOPs strictly lower (sink is free)
- Tests "what if we just didn't compute small at all"

## Cluster

`select=1:ncpus=4:ngpus=1:mem=16gb`, walltime 1h. ~5min actual runtime.

## Outputs

`docs/figures/inference_mask_results.json` — full per-mode per-profile metrics
including routing distribution shifts.
