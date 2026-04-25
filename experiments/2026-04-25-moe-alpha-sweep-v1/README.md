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

| Exp | Job ID | W&B run | α | Status |
|---|---|---|---:|---|
| exp24 | _tbd_ | _tbd_ | 5e-4 | submitted _tbd_ |
| exp25 | _tbd_ | _tbd_ | 1e-3 | submitted _tbd_ |
| exp26 | _tbd_ | _tbd_ | 2e-3 | submitted _tbd_ |
| exp27 | _tbd_ | _tbd_ | 5e-3 | submitted _tbd_ |

## Results

_Pending submission._

| Exp | α | TDLC BLER | UMA BLER | Avg BLER | FLOPs % | Routing (l/n/s tdlc) | Routing (l/n/s uma) |
|---|---:|---:|---:|---:|---:|---|---|

## Decision Criteria

- **Monotone Pareto across 4 points** → headline figure ready, pick winning α
  for 3-seed multi-seed run.
- **Two points dominate same region** → fine, identifies operating range.
- **Any point collapses** → useful negative result, document and bound recipe.
- **Anchor (exp25) ≠ original 3witw8yw within ~2pp BLER** → flag pipeline
  difference; may need to investigate (likely the 50k subset effect).
