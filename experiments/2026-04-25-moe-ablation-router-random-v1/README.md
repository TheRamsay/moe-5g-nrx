# MoE Router Channel-Feature Ablation (v1)

## Question

Do the channel-quality features fed to the router actually drive routing
decisions, or could the same heterogeneous behavior emerge from any router
input?

This tests the central "channel-aware routing" claim of the project.

## Background

The router takes pooled features from the shared stem and produces routing
probs. We claim those features carry channel-quality information that shapes
routing. The skeptic's null: any router input would give the same result —
the architecture (3 experts + Gumbel + load balance + FLOPs penalty) is doing
the work, not channel awareness.

## Configs

| Exp | Router input | Notes |
|---|---|---|
| exp26 | channel_aware (pooled stem features) | Already done — train `t6lkdep2`, eval `2zboo1rh` |
| exp31 | random (fresh Gaussian per forward) | This study |

Identical otherwise: bs=128, 12k steps, asym warm-start, α=2e-3, β=0.1, s67.

Implementation: `model.router.input_mode: random` triggers
`torch.randn(batch, router_input_dim)` in MoENRX.forward instead of pooled
stem features (see `src/models/moe.py`).

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
| exp30 | _tbd_ | _tbd_ | submitted _tbd_ |

## Results — to fill in after eval

| Run | Avg BLER | TDLC routing l/n/s | UMA routing l/n/s | Avg FLOPs % |
|---|---:|---|---|---:|
| exp26 (channel-aware) | 0.902 | 44/15/40 | 26/48/26 | 56% |
| exp30 (random input) | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

## Decision Criteria

| Outcome | Interpretation |
|---|---|
| Random ≈ exp26 BLER (within 0.5 pp) | Channel-aware claim is **wrong**. Architecture/losses do all the work. Reframe story honestly. |
| Random worse BLER, routing collapses | Channel features are load-bearing. **Confirms central claim.** |
| Random worse BLER, routing still heterogeneous | Architecture biases toward heterogeneous routing; channel features improve it. Partial claim. |

Either outcome is publishable. The honest result strengthens the report.
