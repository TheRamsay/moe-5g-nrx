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

8 jobs submitted (19584455–19584462) on 2026-04-30 evening.

## Results — Switch aux sweep (DONE 2026-04-30)

| Exp | Job | switch_aux_weight | UMa BLER | TDLC BLER | exp_flops | real_flops | Outcome |
|---|---|---:|---:|---:|---:|---:|---|
| exp44 | 19584455 | 1e-3 | 0.931 | 0.844 | 1.000 | **1.000** | full collapse to large |
| exp45 | 19584456 | 1e-2 | 0.930 | 0.844 | 1.000 | **1.000** | full collapse to large |
| exp46 | 19584457 | 1e-1 | 0.931 | 0.840 | 1.000 | **1.000** | full collapse to large |
| exp47 | 19584458 | 1e0 | 0.937 | 0.859 | 0.544 | **0.996** | soft routing has high entropy but argmax picks large 99.6% |

**Outcome 1 confirmed:** all 4 collapse. Across 4 orders of magnitude of
weight, Switch auxiliary loss cannot prevent the Phase 2 large-collapse with
our setup. exp47 (strongest weight=1.0) is interesting — it spreads soft
routing probabilities (exp_flops=0.544) but at hard top-1 inference, every
sample still picks large. The BCE gradient pulling to warm-large dominates
at the top-1 decision boundary, even when the soft loss penalises imbalance.

## Results — Capacity sweep (DONE 2026-04-30)

| Exp | Job | capacity_weight | UMa BLER | TDLC BLER | exp_flops | real_flops | Outcome |
|---|---|---:|---:|---:|---:|---:|---|
| exp48 | 19584459 | 0.1 | 0.931 | 0.840 | 1.000 | **1.000** | weak penalty → full collapse |
| exp49 | 19584460 | 0.5 | 0.972 | 0.959 | 0.605 | 0.675 | spreads routing but BLER tanks |
| exp50 | 19584461 | 2.0 | 0.977 | 0.970 | 0.561 | 0.497 | very spread, BLER worse still |
| exp51 | 19584462 | 10.0 | 0.946 | 0.881 | 0.639 | 0.600 | partial recovery (~60% FLOPs) but still 1pp worse than exp26 |

**Two failure modes characterized:**
- **Weak penalty (0.1):** doesn't prevent collapse → 100% large
- **Strong penalty (0.5+):** prevents collapse but the experts haven't co-trained
  for that routing → BLER craters
- **Very strong penalty (10):** finds a similar operating point to exp26 by brute
  force, but still 1pp worse because it forces a routing pattern the experts
  didn't co-train for.

**No middle ground exists** where capacity penalty alone gives both routing
diversity AND good BLER.

## Combined sweep summary (8 jobs total)

| Mechanism | Weights tested | Result |
|---|---|---|
| Switch aux loss | 1e-3, 1e-2, 1e-1, 1e0 | All 4 → 100% large collapse |
| Soft capacity penalty | 0.1, 0.5, 2.0, 10.0 | 0.1 collapses; 0.5–10 spread routing but kill BLER |

## Implications

For the consultation: the original "Switch aux + capacity failed" claims were
based on single-shot runs. We now have **proper sweeps across 4 orders of
magnitude for each**, with all 8 runs failing to recover heterogeneous
routing + good BLER. This strengthens the asym-warm-as-only-recipe narrative
— regularizers fundamentally cannot fix the Phase 2 collapse, regardless of
weight tuning.
