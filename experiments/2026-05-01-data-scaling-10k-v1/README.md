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
