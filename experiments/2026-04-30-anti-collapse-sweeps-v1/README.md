# Phase-2 Anti-Collapse Hyperparameter Sweeps (v1)

## Question

Were the original Phase-2 anti-collapse mechanisms (Switch aux loss + soft
capacity penalty) really tested *thoroughly*, or did we just try one
hyperparameter value each and move on?

If they're properly tuned, can either prevent the Phase 2 v1 collapse to
large that motivated asymmetric warm-start in the first place?

## Background

The original anti-collapse exploration during Phase 1/2 era used **single
training runs** with default hyperparameters:
- Switch aux at default weight (~0.01) — collapsed
- Capacity at one factor — collapsed

This is a real weakness if pressed: "did you tune these?" answer was "no,
single-shot exploration." This sweep fixes that for the two most-cited
mechanisms.

## Configs

All 8 configs use the **Phase 2 v1** base recipe:
- Full warm-start (stem + nano + small + large all from dense checkpoints)
- 2k frozen-experts step (router-only training) → 10k joint
- α=1e-3, β=0.1
- Single seed (67) for sweep efficiency

### Switch aux loss sweep (4 configs)

| Exp | switch_aux_weight |
|---|---:|
| exp44 | 1e-3 |
| exp45 | 1e-2 (Switch paper default) |
| exp46 | 1e-1 |
| exp47 | 1e0 |

Switch aux loss = N · Σ (f_i · P_i) where N=num experts, f_i=fraction routed
to expert i, P_i=mean router probability for expert i. Minimised when routing
is balanced.

### Capacity sweep (4 configs)

| Exp | capacity_weight | capacity_factor |
|---|---:|---:|
| exp48 | 0.1 | 1.5 |
| exp49 | 0.5 | 1.5 |
| exp50 | 2.0 | 1.5 |
| exp51 | 10.0 | 1.5 |

Capacity factor=1.5 → max 50% per expert (3-expert setup). Penalty is
proportional to overflow above the cap.

## Cluster

- Resources: `select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb`
- Walltime: 3h each (12k steps ≈ 40 min on GPU + buffer)
- 8 jobs × 3h = ~24 GPU-hours total

## Submission

```bash
for s in submit_exp4{4,5,6,7}_*.sh submit_exp{48,49,50,51}_*.sh; do
  qsub "$s"
done
```

## Expected outcomes

Three possible patterns:

1. **All 8 collapse to large** → confirms original story: Phase 2 collapse is
   intrinsic to warm-large dominance, no regulariser fixes it. Asym warm-start
   remains the only working recipe. Strong evidence for our narrative.

2. **One or two values work** → "Switch aux at weight=X actually does work" —
   would weaken our claim that asym-warm is *unique*. Honest finding to report.

3. **Mixed**: stronger penalties force uniform routing but kill BLER (like the
   β=2.0 case we already showed). Reinforces "regularizers are crude tools".

## Status

8 jobs submitted (19584455–19584462) on 2026-04-30 evening. Results expected
within 4-6h.
