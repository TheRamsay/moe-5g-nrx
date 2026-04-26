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

| Exp | Train Job | Train W&B | Eval W&B | Status |
|---|---|---|---|---|
| exp31 | 19459438 | 5c0kshem | nyvfkxl0 | done |

## Results (test set, best checkpoint = step 12000)

| Run | TDLC BLER | UMA BLER | **Avg BLER** | TDLC routing | UMA routing | TDLC FLOPs % | UMA FLOPs % |
|---|---:|---:|---:|---|---|---:|---:|
| exp26 (3-expert) | 0.867 | 0.937 | **0.902** | 44/15/40 (l/n/s) | 26/48/26 (l/n/s) | 65% | 47% |
| **exp31 (2-expert)** | **0.878** | **0.940** | **0.909** | 38/62 (l/s) | 37/63 (l/s) | 65% | 65% |

## Verdict — nano earns its keep

Without nano, the router splits roughly 38/62 large/small on both profiles.
BLER worsens by 0.7 pp avg (TDLC: +1.1 pp, UMA: +0.3 pp) and **UMa FLOPs
ratio jumps from 47% to 65%** — small now absorbs the hopeless low-SNR
samples that nano used to handle for ~1/3 the compute, so the average compute
cost climbs.

The TDL-C FLOPs ratio is unchanged (65% both architectures) because TDLC
already used nano sparingly (15%) — the change there is pure BLER, not
compute. The compute story plays out on UMa, where nano was carrying ~half
the routing decisions.

## Decision criteria — outcome

- ✅ **2-expert worse BLER + more FLOPs (on UMa)** → Nano absorbs hopeless
  low-SNR samples. **Justifies the 3-expert design.**

The 3-expert MoE is the better operating point. 2-expert is also a valid
Pareto point but strictly dominated on BLER and on (UMa) FLOPs.

## How this strengthens the report

Without this ablation, a reviewer can ask: "Your sweep showed nano starves
at high α and only gets 15% at the winning α — does it actually contribute?
Why not just use small+large?"

With this ablation, the answer is:

> "We tested the 2-expert (small+large) variant at the same α=2e-3 recipe.
> It performs 0.7 pp worse on BLER and uses 18 pp more FLOPs on UMa, because
> small must now handle low-SNR samples that nano was absorbing more
> efficiently. The 3-expert design is justified."
