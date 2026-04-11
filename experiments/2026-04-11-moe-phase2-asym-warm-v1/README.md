# MoE Phase 2 — Asymmetric Warm-Start v1

## Question

Does removing large's warm-start advantage break the Phase 2 router collapse?

## Background

Phase 2 v1 (exp17, beta=0.1) collapsed: large expert received 100% of traffic
from step 1 and never redistributed. The warm-started large is so much better
than nano/small on every sample that the performance gradient overwhelms both
the FLOPs penalty and load-balance penalty.

All the other anti-collapse experiments in this sprint (beta sweep, capacity
constraint, switch aux) try to *regularize around* the dominance. This run
attacks the root cause: remove the head-start.

- Stem: warm-started (dense_large_s67) — needed for basic competence
- Nano: warm-started (dense_nano_s67)
- Small: warm-started (dense_small_s67)
- **Large: random init** (no warm-start)
- Single joint phase (no freezing): random large must train while router decides
  when to invest in it.

## Config

- Base: `conf/experiment/exp23_moe_phase2_asym_warm.yaml`
- 6k steps, beta=0.1 (matches exp17 Phase 2 v1 for clean A/B)
- freeze_experts=false (single joint phase)
- Walltime: 4h

## Hypotheses

| Outcome | Interpretation |
|---------|----------------|
| Router routes heterogeneously | Warm-start bias was the whole problem. Clean fix. |
| Router collapses to nano/small | Symmetric failure: init quality drives collapse. Problem is deeper. |
| Router collapses to large anyway | Large catches up fast; dominance is architectural not init-dependent. |
| Router initially avoids large, recovers later | Interesting recovery dynamics — worth a second run with longer budget. |

## Jobs

| Job ID | W&B run | Status |
|--------|---------|--------|
| 18937005 | _tbd_ | submitted 2026-04-12 |

## What To Watch

- `train/ema/router_entropy` — want >0.3 throughout (not just at start)
- `train/ema/expert_usage/{nano,small,large}` — track share over time
- `train/ema/expert_usage/large` specifically: does the router ever discover
  large after it trains up?
- `val/tdlc/snr_bin_4/bler` @ SNR=17 — target ≤0.35 (Phase 2 v1 achieved 0.215
  via full collapse; anything comparable with real routing diversity wins)
- `val/tdlc/bler` overall — should not exceed dense large by more than ~0.05

## Results

_Pending job submission._

## Decision Criteria

- **Heterogeneous routing + BLER within 0.05 of dense large** → write this up as
  the primary positive result. Consider extending to 12k steps.
- **Collapse to nano/small** → symmetric result; Phase 1 / Phase 2 / C together
  give a clean 3-point characterization of init-driven collapse. Strong analysis
  contribution even without a working recipe.
- **Collapse to large (large catches up)** → rules out warm-start as sole cause;
  follow up with router regularization (direction B) or asymmetric freezing.
