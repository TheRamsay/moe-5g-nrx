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

| Exp | Train Job | Train W&B | Eval W&B | Status |
|---|---|---|---|---|
| exp30 | 19459437 | cd2w6l31 | ag3qbw52 | done |

## Results (test set, best checkpoint = step 6500)

| Run | TDLC BLER | UMA BLER | **Avg BLER** | TDLC routing l/n/s | UMA routing l/n/s | TDLC FLOPs % | UMA FLOPs % |
|---|---:|---:|---:|---|---|---:|---:|
| exp26 (channel-aware) | 0.867 | 0.937 | **0.902** | 44/15/40 | 26/48/26 | 65% | 47% |
| **exp30 (random input)** | **0.965** | **0.972** | **0.968** | **0/11/89** | **0/11/89** | **41%** | **41%** |

## Verdict — channel-aware features are load-bearing

Random-input router collapses to **always-pick-small** (0% large, 11% nano,
89% small on both profiles) and BLER craters by **6.6 pp avg** (TDLC: +9.8 pp,
UMA: +3.5 pp). Without channel-quality information the router cannot decide
when to invest in the expensive large expert — it defaults to the
"cheap-but-OK" option, which is correct on UMa (mostly easy) but catastrophic
on TDL-C (where waterfall samples need large).

Best checkpoint at step 6500 (model never recovered). Earlier training-EMA
snapshots showed ~33/33/33 routing because the fresh-noise input gives
roughly uniform argmax — but the FLOPs penalty pulled the actual policy
toward small over training.

## Decision criteria — outcome

- ✅ **Random worse BLER, routing collapses** → Channel features are
  load-bearing. **Central claim confirmed.**

This is the strongest possible outcome of the ablation. It directly supports
the "compute-aware MoE uses channel-quality features to decide where to
spend FLOPs" framing. The contrast — same architecture, same losses, same
warm-start, only the router input changed — isolates channel-aware as the
operative variable.

## How this strengthens the report

Previously the channel-aware claim was framed as "the router takes pooled
stem features." With this ablation we can now say:

> "We confirmed via ablation that channel-quality features in the router
> input are load-bearing for compute-aware routing: replacing them with
> Gaussian noise causes the router to collapse onto a single cheap expert
> and BLER to degrade by 6.6 pp on average."
