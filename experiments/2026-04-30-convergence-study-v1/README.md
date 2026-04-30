# Convergence Study — exp26 at 30k Steps (v1)

## Question

We use 12k training steps for all MoE ablations and sweeps (fair A/B
comparison budget). Is 12k converged for the asym-warm recipe, or does
training longer meaningfully improve BLER?

## Background

Standard ML methodology has three training tiers:
1. Ablations / sweeps: fixed budget (12k for us) for fair comparison
2. Headline / final model: trained to convergence
3. Multi-seed for confidence intervals

We've done (1) thoroughly. We've done (3) at 50k samples. We have NOT done
(2) — exp26's 0.902 BLER is the 12k-step number, not the converged number.

Existing evidence: a single 20k extension (CLAUDE.md line 158) showed val
TDLC BLER dropping from 0.867 (12k) → 0.851 (16k). So 12k is mildly
under-converged.

## Configs

| Exp | Setup | Notes |
|---|---|---|
| exp26 (existing) | alpha=2e-3, asym warm, seed 67, **12k steps** | Headline ablation number |
| **exp59 (this)** | identical recipe, **30k steps**, val every 500 steps | Convergence study + final headline |

## Cluster

- Resources: `select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb`
- Walltime: 6h (12k = ~40 min compute; 30k = ~100 min + buffer)

## Predicted outcomes

- Conservative: BLER drops from 0.902 (12k) to ~0.88-0.89 (30k). Routing
  stabilizes around step 16-18k. Confirms 12k is under-converged but not
  catastrophically so. Final headline number for the report: ~0.89.
- Optimistic: BLER drops to ~0.86-0.87 (close to dense_large's 0.901 ÷
  whatever advantage it had). Big visual win.
- Surprising: BLER doesn't improve much, possibly because the recipe hits
  a local minimum that isn't escaped by more training.

## Status

Submitted 2026-04-30 evening. ~3-5h training expected.

## After this finishes

If results are promising, the proper rigor for the final report would be:
- Re-run exp59 recipe with seeds 32 and 42 (multi-seed at 30k)
- ~3 jobs, would land before deadline if started early next week
