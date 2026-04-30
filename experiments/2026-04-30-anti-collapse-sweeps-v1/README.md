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

**Capacity sweep (exp48-51) in flight at time of writing.**

## Implications

For the consultation: the original "Switch aux failed" claim was based on a
single-shot run. We now have a **proper sweep across 4 orders of magnitude**
that confirms the result. This strengthens the asym-warm-as-only-recipe
narrative — Switch aux fundamentally cannot fix Phase 2 collapse, regardless
of weight tuning.
