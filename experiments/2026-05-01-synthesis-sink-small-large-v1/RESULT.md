# exp64 RESULT — Total collapse to sink

## Verified facts (W&B summary `19595889`)

| Metric | Value | Source |
|---|---:|---|
| val/uma/bler | **1.000000** | `wandb-summary.json` |
| val/tdlc/bler | **1.000000** | `wandb-summary.json` |
| val loss | 0.7218 | `wandb-summary.json` (≈ ln(2)) |
| train/profile/uma/ema/expert_usage/sink | **1.0000** | `wandb-summary.json` |
| train/profile/tdlc/ema/expert_usage/sink | **1.0000** | `wandb-summary.json` |
| train/profile/{uma,tdlc}/ema/expert_usage/{small,large} | **0.0000** | `wandb-summary.json` |
| train/ema/realized_flops_ratio | 0.1776 | `wandb-summary.json` (just stem) |

## Trajectory

Verified from `run.log`: collapsed to all-sink at step 500 and stayed there
until step 12000.

| Step | UMa BLER | TDLC BLER |
|---|---:|---:|
| 500 | 1.0000 | 1.0000 |
| 1000 | 1.0000 | 1.0000 |
| 1500 | 1.0000 | 1.0000 |
| 11000 | 1.0000 | 1.0000 |
| 12000 | 1.0000 | 1.0000 |

The val loss being EXACTLY 0.7218 ≈ ln(2) = 0.693 is the smoking gun:
sink outputs zeros, so each bit's BCE is ln(2) per bit. Constant across all
samples, all SNRs, all profiles → confirms 100% sink routing.

## Mechanism

**Sink at 0 FLOPs is too attractive under α=2e-3.** The FLOPs-penalty
gradient pushes the router to all-sink before warm-started small/large can
demonstrate value. The router converges within ~500 steps and never
recovers.

For comparison, exp26's nano (the cheapest decoder) costs 320M FLOPs.
That nonzero compute keeps the FLOPs penalty from completely dominating —
the router has incentive to occasionally try other experts.

## Implication

**Sink-as-an-expert requires different training dynamics than the asym-warm
recipe was tuned for.** Possible recipe modifications (NOT tested here):
- α-warmup (start at 0, ramp up)
- Higher load-balance β to force exploration
- Higher Gumbel temperature for longer
- Dedicated sink-warmup phase

Out of scope for this work — the inference-time substitution (mode B)
achieves the same goal without touching training.

## Architectural verdict

exp64 is a clean negative result. Sink belongs at INFERENCE, not at TRAINING.
See `experiments/2026-05-01-inference-mask-v1/RESULT.md` for the working
mode B recipe.
