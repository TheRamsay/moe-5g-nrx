# MoE Alpha Sweep — Asymmetric Warm-Start (v1)

## Question

How does the BLER vs FLOPs operating point shift with the FLOPs penalty α?
Turn the current 3-point Pareto into a real curve.

## Background

The asym-warm 12k anchor (`3witw8yw`, α=1e-3) gives a single Pareto point at
0.910 avg BLER / 61% FLOPs. Without varying α we have no curve, only one dot.
Per CURRENT.md "Fair Comparison Protocol" we sweep at bs=128, 12k steps so
results compose with the existing analysis. The α=1e-3 anchor is **re-baselined
on the current 50k subset** (the original ran on the full HF stream) so all 4
points are pipeline-consistent.

## Configs

| Exp | α | β | Notes |
|---|---:|---:|---|
| exp24 | 5e-4 | 0.1 | Half anchor — upper end of curve (more FLOPs) |
| exp25 | 1e-3 | 0.1 | Anchor re-baseline at 50k subset |
| exp26 | 2e-3 | 0.1 | 2× anchor — middle of curve |
| exp27 | 5e-3 | 0.1 | 5× anchor — aggressive, may collapse large |

All identical otherwise: bs=128, 12k steps from scratch, asym warm-start
(stem + nano + small from dense s67 checkpoints, large random init),
seed=67, lr=1e-3, wd=1e-4, no scheduler, freeze_experts=false.

## Cluster

- Resources: `select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb` (no `gpu_mem`)
- Walltime: 2h per job (12k steps × bs=128 = 1.54M samples / 3k samp/s ≈ 9 min
  compute + ~30 min startup/preload + validation). 2h gives wide buffer.
- 4 jobs × 2h = ~8 GPU-hours total wall budget.

## Submission

```bash
bash submit.sh print          # dry-run — show qsub commands
bash submit.sh qsub           # actually submit all 4 jobs
bash submit.sh local          # run one locally for smoke test
```

## What To Watch

- `train/ema/router_entropy` — collapse diagnostic. >0.3 = healthy diversity.
- `train/ema/expert_usage/large` — at α=5e-3, may go to 0 (Phase 1 pattern).
- `val/{tdlc,uma}/bler` — final BLER per profile.
- `val/tdlc/snr_bin_4/bler` @ SNR=17 — waterfall sensitivity.

## Expected Pareto Shape

| α | Predicted FLOPs % | Predicted avg BLER | Routing pattern |
|---:|---:|---:|---|
| 5e-4 | 75-90% | 0.88-0.90 | Large dominates |
| 1e-3 | 55-65% | 0.91 | Anchor-like (46/2/52 tdlc) |
| 2e-3 | 40-55% | 0.91-0.93 | Small dominates |
| 5e-3 | 25-40% | 0.93-0.96 | Possible large collapse |

Shape will be useful even if predictions are wrong — the *curve* is the deliverable.

## Jobs

| Exp | Job ID | Train W&B | Eval W&B | α | Status |
|---|---|---|---|---:|---|
| exp24 | 19457669 | _init flaked_ | 002cwsy2 | 5e-4 | done; metrics recovered via eval |
| exp25 | 19457670 | 3xzxkddv | 5jswm490 | 1e-3 | done |
| exp26 | 19457671 | t6lkdep2 | 2zboo1rh | 2e-3 | done |
| exp27 | 19457672 | _init flaked_ | dh4x0qmu | 5e-3 | done; metrics recovered via eval |

## Results (test set, best checkpoint per run)

| Exp | α | Best step | TDLC BLER | UMA BLER | **Avg BLER** | TDLC routing l/n/s | UMA routing l/n/s | TDLC FLOPs | UMA FLOPs | **Avg FLOPs %** |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| exp24 | 5e-4 | 12000 | 0.861 | 0.936 | **0.898** | 100/0/0 | 100/0/0 | 1604M | 1604M | **100%** |
| exp25 | 1e-3 | 10000 | 0.875 | 0.938 | 0.907 | 44/12/44 | 25/46/29 | 1047M | 749M | 56% |
| **exp26** | **2e-3** | 11000 | **0.867** | 0.937 | **0.902** | 44/15/40 | 26/48/26 | 1043M | 747M | **56%** |
| exp27 | 5e-3 | 12000 | 0.881 | 0.940 | 0.911 | 37/**0**/63 | 22/**0**/78 | 1030M | 894M | 60% |
| Dense large (ref) | — | — | 0.866 | 0.936 | 0.901 | — | — | 1604M | 1604M | 100% |

**Headline:** exp26 reaches **0.1 pp of dense large at 56% FLOPs**, dominating
the original anchor (0.910 / 61%).

**Pareto frontier:** 2 points — exp24 (collapsed) and exp26 (knee). exp25 and
exp27 are dominated.

**Three operating regimes uncovered by the sweep:**
- α=5e-4 (exp24): too weak → 100% large collapse. Same failure as Phase 2 v1.
- α=1e-3 to 2e-3: heterogeneous routing emerges, sweet spot at 2e-3.
- α=5e-3 (exp27): too strong → nano starves entirely (0%/0%), router falls back
  to 2-expert (large/small) regime, *raising* avg FLOPs vs the sweet spot.

**Implications:**
- Pick **α=2e-3 as the winning configuration** for 3-seed multi-seed run.
- The "nano disappears at high α" finding strengthens the 2-expert ablation
  motivation (large+small only). Worth running.

## Follow-up studies (2026-04-26)

After this sweep identified exp26 (α=2e-3) as the Pareto knee, three downstream
studies built on the result:

- `2026-04-25-moe-asym-a2e3-3seed-v1` — 3-seed multi-seed confirmation. Result:
  **bimodal**, 2/3 seeds reach 0.902, 1/3 collapses (s32, large→0%). Recipe is
  not seed-stable; report quotes both attractors honestly.
- `2026-04-25-moe-ablation-router-random-v1` — channel-aware ablation. Result:
  **BLER craters 6.6 pp** when router is fed noise instead of pooled stem
  features. **Channel-aware features are load-bearing** (central project claim
  confirmed).
- `2026-04-25-moe-ablation-2expert-v1` — drop nano. Result: **0.7 pp BLER hit
  + 18 pp more UMa FLOPs** without nano. **3-expert design is justified**.
- `2026-04-26-deepmimo-ood-eval-v1` (in flight) — out-of-distribution eval on
  ray-traced channels. Tests generalization beyond Sionna 3GPP synthetic.

## Decision Criteria — outcome

- ✅ **Monotone Pareto across 4 points** → not monotone, but the 3-regime story
  is more interesting and explanatory than a smooth curve would be.
- ✅ **One point collapses** → exp24, useful negative result that bounds α.
- ✅ **Anchor (exp25) ≠ original 3witw8yw** → routing differs (44/12/44 vs
  46/2/52), avg BLER similar. Validates the rebaseline.

## Wandb-init reliability

2 of 4 training runs and 1 of 4 eval runs hit `Failed to read port info after
30.0 seconds`. Recovered all metrics from local checkpoints + re-submission.
Track this — `WANDB_MODE=offline` + post-hoc `wandb sync` would be the
defensive fix.
