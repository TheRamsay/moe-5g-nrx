# MoE Asym-Warm α=2e-3 — 3-Seed Confirmation

## Question

Is the alpha-sweep winner (exp26, α=2e-3, avg BLER 0.902 / 56% FLOPs) a stable
result across seeds, or did seed=67 land in a lucky basin?

## Background

The 4-point alpha sweep (`2026-04-25-moe-alpha-sweep-v1`) identified α=2e-3
as the Pareto knee: 0.1 pp from dense large baseline at 56% FLOPs. Single
seed (s67). Per Research-positioning §9, the headline result needs ≥3 seeds.

If all 3 seeds give heterogeneous routing and BLER within ~0.5 pp of each
other, the headline holds. If 1 seed collapses or gives wildly different
routing, the recipe has an instability worth flagging.

## Configs

Identical to exp26 in every variable except seed.

| Exp | seed | Notes |
|---|---:|---|
| exp26 | 67 | Already done (sweep) — train run `t6lkdep2`, eval `2zboo1rh` |
| exp28 | 32 | This study |
| exp29 | 42 | This study |

All: bs=128, 12k steps, asym warm-start (stem+nano+small from dense s67
checkpoints, large random init), α=2e-3, β=0.1, lr=1e-3, wd=1e-4, no scheduler.

## Cluster

- Resources: `select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb` (no `gpu_mem`)
- Walltime: 2h per job (24 min preload + 17 min train at 12k×128 / 1.4k samp/s + val + cleanup)
- 2 jobs in parallel (s32 + s42), s67 already done.

## Submission

```bash
bash submit.sh print          # dry-run
bash submit.sh qsub           # submit s32 + s42
```

## Jobs

| Exp | seed | Train Job | Train W&B | Eval W&B | Status |
|---|---:|---|---|---|---|
| exp28 | 32 | 19459404 | _wandb-init flaked, ckpt local-only_ | ywvyzlia | done |
| exp29 | 42 | 19459405 | 121ex9e6 | zwwkc1mg | done |

## Results (test set, best checkpoint per run)

| Seed | Best step | TDLC BLER | UMA BLER | **Avg BLER** | TDLC routing l/n/s | UMA routing l/n/s | TDLC FLOPs % | UMA FLOPs % |
|---:|---:|---:|---:|---:|---|---|---:|---:|
| 67 (exp26) | 11000 | 0.867 | 0.937 | **0.902** | 44/15/40 | 26/48/26 | 65% | 47% |
| 42 (exp29) | 7500 | 0.868 | 0.937 | **0.902** | 55/12/33 | 31/42/27 | 71% | 51% |
| **32 (exp28)** | **11000** | **0.949** | **0.967** | **0.958** | **0/49/51** | **0/46/54** | **32%** | **33%** |

**Bimodal outcome** — two of three seeds reproduce the headline; one collapsed.

- **s67 + s42** (the "good" attractor): heterogeneous routing, large gets
  44–55% on TDLC, BLER 0.902 within 0.1 pp of dense large. Routing patterns
  differ in detail (s67 uses small more, s42 uses large more) but operating
  point is the same.
- **s32** (the "collapsed" attractor): large completely starved (0% on both
  profiles), routing splits roughly 50/50 between nano and small. BLER is
  5.6 pp worse but FLOPs are ~half. This is the same Phase-1 failure mode but
  reached from the asym-warm recipe.

**Mean ± std (excluding the s32 outlier):** 0.902 ± 0.000.
**Including s32:** 0.921 ± 0.032 (but two distinct attractors, so a single
mean misrepresents the result).

## Verdict

The α=2e-3 winning configuration is **not seed-stable** — the asym-warm
recipe has a real bimodal failure mode. Honest finding for the report:
> "Of 3 random seeds (32, 42, 67), 2 reach 0.902 avg BLER at 56–65% FLOPs
> with heterogeneous routing; the third collapsed to a 2-expert nano/small
> regime (0.958 BLER, 33% FLOPs). The recipe's success rate is therefore
> ~67%, suggesting room for follow-up work on stabilization (e.g., gradual
> warmup of large, or capacity constraints during the random-init phase)."

This is a stronger result than "3 seeds all gave the same number" because it
characterizes **where the recipe breaks** rather than hiding the variance.
