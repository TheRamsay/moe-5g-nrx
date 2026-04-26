# MoE β-Warmup Stabilization (v1)

## Question

Does heavy load-balance pressure during the random-init phase (β=0.5 for
first 4k steps, then β=0.1) stabilize the asym-warm recipe **without**
over-correcting like the large-warmup did?

## Background

Two failure modes characterized so far:

| Recipe | Outcome | Failure mode |
|---|---|---|
| exp26 (no warmup) | 2/3 seeds heterogeneous, 1/3 (s32) collapses | Large too weak too long → router avoids it |
| exp32+33 (large-warmup, 2k frozen nano+small) | All seeds collapse to 100% large | Large too strong by step 2000 → Phase 2 v1 collapse |

The sweet spot is between. β-warmup attempts a different mechanism: instead
of freezing experts, force routing diversity via strong load-balance
penalty during the vulnerable phase, then relax.

## Configs

| Exp | seed | β schedule | Notes |
|---|---:|---|---|
| exp35 | 67 | 0.5 → 0.1 at step 4000 | Anchor seed |
| exp36 | 32 | 0.5 → 0.1 at step 4000 | **Critical seed** — collapsed at exp26 |
| exp37 | 42 | 0.5 → 0.1 at step 4000 | Test seed |

Code change: `src/training/trainer.py` `_compute_loss` now supports
`load_balance_beta_warmup_steps` + `load_balance_beta_warmup_value`. When
both set, β = warmup_value while global_step < warmup_steps, then steady-state.

Otherwise identical to exp26: bs=128, 12k steps, asym warm-start, α=2e-3.

## Cluster

- 3 jobs, ~45 min each
- `select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb`

## Submission

```bash
bash submit.sh print
bash submit.sh qsub
```

## Jobs

| Exp | seed | Job ID | Train W&B | Eval W&B | Status |
|---|---:|---|---|---|---|
| exp35 | 67 | _tbd_ | _tbd_ | _tbd_ | submitted _tbd_ |
| exp36 | 32 | _tbd_ | _tbd_ | _tbd_ | submitted _tbd_ |
| exp37 | 42 | _tbd_ | _tbd_ | _tbd_ | submitted _tbd_ |

## Decision Criteria

| Outcome | Story |
|---|---|
| **All 3 seeds reach heterogeneous routing + ≤0.92 BLER** | Recipe stabilized. **β-warmup is the new default.** Headline upgrades to "0.902 across 3 seeds." |
| 3 collapse to 100% large | β=0.5 not strong enough to overcome large's pull after step 4000. Try β=1.0 in warmup. |
| 3 collapse to 0% large | β too strong even after warmup. Try shorter (β=0.5 for 2k steps). |
| Mixed across seeds (like exp26) | β-warmup ≈ no-op. Bimodality has a different cause. |
