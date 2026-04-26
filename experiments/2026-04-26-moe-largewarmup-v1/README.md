# MoE Large-Warmup Stabilization (v1)

## Question

Can a 2k-step **frozen-large warmup** prevent the asym-warm recipe's bimodal
collapse (1 of 3 seeds at exp26 collapsed to 0% large)?

## Background

`2026-04-25-moe-asym-a2e3-3seed-v1` showed that exp26's α=2e-3 recipe is
**not seed-stable**: 2 of 3 seeds (s67, s42) reach the 0.902 BLER /
heterogeneous-routing attractor; 1 of 3 (s32) collapses to 0/49/51
(large→0, BLER 0.958). Same Phase-1 large-aversion failure, just from a
different seed.

Hypothesis: large is too weak too early. Even with asym warm-start, large
spends the first ~6k steps catching up to nano+small (which are warm-started).
For some seeds, the FLOPs penalty (α=2e-3) pushes the router away from
large during this vulnerable window, and it never recovers.

**Stabilization recipe:** freeze nano + small for the first 2k steps so only
stem + large + router train. By step 2000, large should be roughly competitive
with nano/small. Then unfreeze nano+small for joint training (10k more
steps).

Code change: `src/training/trainer.py` `_apply_freeze_config` now supports
`freeze_experts_list: [nano, small]` (named subset) in addition to the
all-or-nothing `freeze_experts: true`.

## Configs

| Exp | seed | Notes |
|---|---:|---|
| exp32 | 67 | Test seed (worked at exp26) |
| exp33 | 32 | **Critical seed** — collapsed at exp26. Recipe must rescue this one. |
| exp34 | 42 | Test seed (worked at exp26) |

All three: bs=128, 12k steps, asym warm-start (stem+nano+small from dense
checkpoints, large random), `freeze_experts_list=[nano,small]`,
`unfreeze_experts_at_step=2000`, α=2e-3, β=0.1, lr=1e-3.

## Cluster

- 3 jobs, ~45 min each (preload + train + val)
- Resources same as exp26: `select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb`

## Submission

```bash
bash submit.sh print
bash submit.sh qsub
```

## Jobs

| Exp | seed | Job ID | Train W&B | Eval W&B | Status |
|---|---:|---|---|---|---|
| exp32 | 67 | _tbd_ | _tbd_ | _tbd_ | submitted _tbd_ |
| exp33 | 32 | _tbd_ | _tbd_ | _tbd_ | submitted _tbd_ |
| exp34 | 42 | _tbd_ | _tbd_ | _tbd_ | submitted _tbd_ |

## Results — to fill in after eval

| Seed | Best step | TDLC BLER | UMA BLER | Avg BLER | TDLC routing l/n/s | UMA routing l/n/s | TDLC FLOPs % |
|---:|---:|---:|---:|---:|---|---|---:|
| 67 (exp32) | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| 32 (exp33) | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| 42 (exp34) | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

## Decision Criteria

| Outcome | Story update |
|---|---|
| **All 3 seeds reach heterogeneous routing + ≤0.92 BLER** | **Recipe stabilized.** Frozen-large warmup is the new default. Headline upgrades from "0.902 ± large variance" to "0.902 across 3 seeds." |
| 2 of 3 stabilize, s32 still collapses | Partial fix. The s32 init is just pathological. Quote both attractors. |
| All 3 collapse to a different attractor (e.g., always-large) | Warmup too aggressive — large now dominates routing. Try shorter warmup (1k steps). |
| All 3 reach exp26-like result | Warmup is no-op; bimodality has a different cause. Try other recipes (α schedule, β warmup). |
