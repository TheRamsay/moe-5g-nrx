# exp65 RESULT — Training instability

## Verified facts (W&B summary `19596144`)

| Metric | Value | Source |
|---|---:|---|
| val/uma/bler | **0.976196** | `wandb-summary.json` |
| val/tdlc/bler | **0.966919** | `wandb-summary.json` |
| **avg val BLER** | **0.9716** | computed |
| Δ vs exp26 (0.9021) | **+6.95pp WORSE** | |
| train/ema/realized_flops_ratio | 0.5827 | `wandb-summary.json` |
| Routing TDLC EMA: sink / nano / large | 0.6% / 47.5% / 51.9% | `train/profile/tdlc/ema/...` |
| Routing UMa EMA: sink / nano / large | 34.0% / 28.9% / 37.2% | `train/profile/uma/ema/...` |

## Trajectory

Verified from `run.log` — partial training, then drifted back UP:

| Step | UMa BLER | TDLC BLER | Note |
|---|---:|---:|---|
| 500 | 0.9999 | 1.0000 | warm-up |
| 1000 | 0.9584 | 0.9348 | improving |
| 1500 | 0.9539 | 0.9169 | improving |
| 11000 | 0.9816 | 0.9760 | drifted back up |
| 11500 | 0.9785 | 0.9739 | unstable |
| 12000 | 0.9762 | 0.9669 | final |

## Asymmetric sink usage

Note the striking asymmetry:
- **TDLC: sink 0.6%** (almost never used)
- **UMa: sink 34.0%** (used heavily)

The router treats UMa samples as more often "hopeless" (1/3 routed to sink)
while TDLC samples are routed only to nano/large. This matches the
profile-difficulty hypothesis: UMa channels are more often unrecoverable.

But because nano alone has to cover both "random output for hard samples"
AND "partial decode for borderline samples" without a separate small tier,
the routing is unstable.

## Mechanism

Different failure mode than exp64 (full sink-collapse):
- exp64 had warm-started **small** which gave smooth BCE early; sink dominated by FLOPs
- exp65 has warm-started **nano** (much weaker decoder); sink wasn't immediately better
- Result: router used nano + large + some sink, but couldn't develop a stable policy

## Architectural verdict

Combined with exp64 (full collapse) and exp61 v2 (FLOPs +22pp): **all three
sink-in-architecture attempts failed.** The recipe is fundamentally not
sink-friendly. The only working sink approach is inference-time substitution
(see `experiments/2026-05-01-inference-mask-v1/RESULT.md` for mode B).
