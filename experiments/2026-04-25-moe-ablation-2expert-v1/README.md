# MoE 2-Expert Ablation — Drop Nano (v1)

## Question

Does nano contribute meaningfully, or is small+large sufficient?

## Background

Sweep findings (`2026-04-25-moe-alpha-sweep-v1`):
- α=2e-3 (winner): nano gets 15% TDLC / 48% UMA
- α=5e-3: nano starves entirely (0%/0%) — router prefers small even with
  stronger FLOPs penalty
- This suggests nano's BLER cost may outweigh its FLOPs savings vs small.

The 2-expert (small+large) variant tests whether the 3rd expert is justified.

## Configs

| Exp | Experts | Notes |
|---|---|---|
| exp26 | nano + small + large | Already done — train `t6lkdep2`, eval `2zboo1rh` |
| exp31 | small + large (no nano) | This study |

Identical otherwise: bs=128, 12k steps, asym warm-start (stem+small warm,
large random), α=2e-3, β=0.1, s67.

## Cluster

- Resources: `select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb`
- Walltime: 2h (same as exp26)
- 1 job.

## Submission

```bash
bash submit.sh print
bash submit.sh qsub
```

## Jobs

| Exp | Job ID | W&B run | Status |
|---|---|---|---|
| exp31 | _tbd_ | _tbd_ | submitted _tbd_ |

## Results — to fill in after eval

| Run | Avg BLER | TDLC routing | UMA routing | Avg FLOPs % |
|---|---:|---|---|---:|
| exp26 (3-expert) | 0.902 | 44/15/40 (l/n/s) | 26/48/26 (l/n/s) | 56% |
| exp31 (2-expert) | _tbd_ | _tbd_ (l/s) | _tbd_ (l/s) | _tbd_ |

## Decision Criteria

| Outcome | Interpretation |
|---|---|
| 2-expert ≈ 3-expert BLER & FLOPs | Nano is dead weight. Update arch story to "2 heterogeneous experts is enough." |
| 2-expert worse BLER, similar FLOPs | Nano absorbs hopeless low-SNR samples. **Justifies 3-expert design.** |
| 2-expert lower FLOPs, slightly worse BLER | Real Pareto tradeoff — both architectures valid. |

Strengthens the architecture justification either way.
