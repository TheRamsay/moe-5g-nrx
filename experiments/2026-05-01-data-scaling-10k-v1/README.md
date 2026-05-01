# Data Scaling — 10k Lower-Bound Test

## Question

Teacher's concern was "50k might be too small." We answered the upper bound
with exp60 (100k matches 50k under α/data scaling). This brackets the LOWER
bound: **does the headline recipe still produce heterogeneous routing at 10k?**

## Hypothesis

Two outcomes both inform the report:

| Outcome | Interpretation |
|---|---|
| exp63 succeeds at α=2e-3 | Recipe is robust across {10k, 50k, 100k} — even nicer story |
| exp63 collapses at α=2e-3 | α/data principle holds bidirectionally; need exp64 with α=5e-3 |

Either way we close the data-scaling story.

## Recipe

Same as exp26: α=2e-3, asym warm-start, seed 67, 12k steps. Only difference:
`hf_max_samples=10000` (1/5 of headline 50k).

Reuses cached `train-50k-array3d` directory — Hydra's loader truncates to 10k.

## Cluster

`select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb`, walltime 3h.

## Result (DONE 2026-05-01, job 19594871) — verified from W&B summary

| Metric | exp26 (50k, train EMA) | **exp63 (10k, train EMA)** | Δ |
|---|---:|---:|---|
| UMa BLER (val) | 0.941 | **0.943** | +0.002 |
| TDLC BLER (val) | 0.864 | **0.868** | +0.004 |
| **Avg BLER** | **0.902** | **0.906** | +0.4pp |
| UMa FLOPs ratio | 0.46 | 0.66 | +20pp |
| TDLC FLOPs ratio | 0.66 | 0.74 | +8pp |
| **Avg FLOPs ratio** | **0.57** | **0.71** | **+14pp** |

**Routing distribution (train EMA):**

| | TDLC l/s/n | UMa l/s/n |
|---|---|---|
| exp26 | 42% / 39% / 19% | 25% / 39% / 36% |
| **exp63** | **55% / 45% / 0%** | **41% / 60% / 0%** |

**Outcome: BLER-robust, routing-policy degrades.** With only 10k samples, the router
COMPLETELY ABANDONED nano (0% on both profiles) and collapsed to a 2-tier {small, large}
policy. BLER stays within 0.4pp of exp26, but FLOPs increases +14pp because the cheap
"hopeless detector" (nano) tier is missing.

## Methodological finding

> *"The asym-warm recipe is BLER-robust to data scale but FLOPs-fragile to data scale.
> With only 10k samples the router cannot develop fine-grained 3-tier routing — it
> collapses to a coarser 2-tier {small, large} policy and forfeits ~14pp of the FLOPs
> benefit. Combined with exp40/58 (100k+α=2e-3 collapse) and exp60 (100k+α=1e-3 recovers),
> the picture is: data scale and FLOPs penalty α must both be in the right range — too
> little data and the router can't learn nano specialization; too high α at large data
> and the router collapses to nano-heavy."*

## Verification

All numbers above pulled directly from
`/storage/brno2/home/ramsay/moe-5g-nrx/results/19594871.pbs-m1.metacentrum.cz/wandb/wandb/run-*/files/wandb-summary.json`
on 2026-05-01. Comparison values for exp26 from
`results/19457671.pbs-m1.metacentrum.cz/wandb/...` for fair train-EMA comparison
(exp26 also has eval-set values that match within rounding: UMa 0.937 / TDLC 0.867 / FLOPs 56%).
