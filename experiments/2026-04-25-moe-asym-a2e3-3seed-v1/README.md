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

| Exp | seed | Job ID | Train W&B | Status |
|---|---:|---|---|---|
| exp28 | 32 | _tbd_ | _tbd_ | submitted _tbd_ |
| exp29 | 42 | _tbd_ | _tbd_ | submitted _tbd_ |

## Results — to fill in after eval

3 seeds total (s67 from exp26, s32 from exp28, s42 from exp29).

| seed | Best step | TDLC BLER | UMA BLER | Avg BLER | TDLC routing l/n/s | UMA routing l/n/s | Avg FLOPs % |
|---:|---:|---:|---:|---:|---|---|---:|
| 67 | 11000 | 0.867 | 0.937 | 0.902 | 44/15/40 | 26/48/26 | 56% |
| 32 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| 42 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| **mean ± std** | | | | _tbd_ | | | _tbd_ |

## Decision Criteria

- **All 3 seeds within ~0.5 pp BLER + heterogeneous routing** → headline holds.
  Quote `mean ± std` in CLAUDE.md / report.
- **One seed collapses to large or to nano-starvation** → asym-warm recipe has
  an instability. Run a 4th seed; report whichever attractor is dominant.
- **Routing patterns diverge wildly across seeds** → the SNR-adaptive routing
  story is seed-dependent. Reframe as "one of multiple attractors found".

## After this study

Update CLAUDE.md / CURRENT.md / checkpoint_report with `mean ± std`. Then
move on to:
- Random-feature router ablation (proves channel-aware claim)
- 2-expert ablation (motivated by sweep finding that nano starves at high α)
- DeepMIMO OOD eval
