# Function-Specialized MoE — Sink + Channel-Only + Decoder (v1) — CONFOUNDED

> **⚠️ Hydra config bug:** v1 (job 19588534) silently kept moe_nl's
> nano + small experts via deep-merge alongside the new sink/channel_only/large.
> Instantiated model had **5 experts**, not 3. Results below are valid for a
> 5-expert architecture but do NOT cleanly test the function-specialized hypothesis.
>
> **Clean version: see `2026-05-01-function-specialized-moe-v2/`** (job 19594735).
>
> Checkpoint inspection of v1 confirmed the bug:
> ```
> nano:         10,220 params  ← from moe_nl, should NOT be here
> small:        115,468 params  ← from moe_nl, should NOT be here
> large:        369,868 params
> channel_only: 109,608 params
> sink:         (0 params)
> Total:        692k (vs intended 567k)
> ```
>
> **v1 result:** avg BLER ~0.911, real_flops ~0.35 (5-expert).

## Question

Our per-expert success-rate analysis (2026-04-30) revealed:
- nano: **0.00%** decode success on its routed samples
- small: **0.00%** decode success on its routed samples
- large: 23-29% success

**Why give nano and small full decoder heads if they never decode?** Replace
with function-specialized experts:

1. **sink** — zero parameters, returns zeros for both bits and channel.
   For samples where decoding is hopeless. Saves 100% of expert FLOPs.
2. **channel_only** — backbone + channel readout only (no bit-LLR head).
   For samples where channel is learnable but bits aren't. Still
   contributes to the auxiliary channel-MSE loss.
3. **decoder (large)** — full bit + channel heads. Unchanged from exp26.

## Code changes

`src/models/moe.py`:
- Added `_SinkExpert` (zero parameters, zeros output)
- Added `_ChannelOnlyExpert` (backbone + channel readout, zero logits)
- `MoENRX.__init__` dispatches expert type via `experts_config[name]['type']`
- `_estimate_expert_flops` accounts for sink (0) and channel-only (no LLR head)

`src/models/warm_start.py`:
- Made expert load lenient — filter checkpoint keys to those present in target
  expert. Allows warm-starting channel_only from dense_small (skipping the
  bit-LLR head weights).

## Architecture comparison

| Expert | Params | FLOPs (expert-only) | What it produces |
|---|---:|---:|---|
| **exp26** nano | 90k | 35M | bits + channel (but never decodes) |
| **exp26** small | 168k | 410M | bits + channel (but never decodes) |
| **exp26** large | 450k | 1319M | bits + channel (decodes) |
| **exp61** sink | 0 | 0 | zeros (skip-the-compute) |
| **exp61** channel_only | ~150k | ~310M | channel only (no bit head) |
| **exp61** large | 450k | 1319M | bits + channel (decodes) |

## Recipe

Same exp26 base: alpha=2e-3, asym warm-start, seed 67, 12k steps.
- Sink: no warm-start (no params)
- Channel_only: warm-start from dense_small (loads backbone + channel readout, skips bit head)
- Large: random init (asym warm-start)

## Cluster

- Resources: `select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb`
- Walltime: 3h

## Predicted outcomes

**Best case:** matches exp26 BLER (~0.902) at lower FLOPs because sink is
much cheaper than nano. **Total compute could drop from 56% to ~40%.**

**Failure modes:**
- Routing collapses (router can't find the right pattern with new architecture)
- BLER suffers because channel-only's lack-of-bit-head changes gradient flow
- Asym warm-start dynamics differ with the new expert types

**Either outcome publishable** — confirms or refines the "compute optimizer"
interpretation of nano/small.
