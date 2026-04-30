# Small-Expert Ablations — Symmetric to exp31 + Smaller-Small (v1)

## Question

Two related questions raised post-consultation:

1. **Is the small expert just a sink?** exp31 dropped nano (kept small+large)
   and got worse — but we never tested the symmetric case. If small mostly
   absorbs hopeless low-SNR samples, then `{nano, large}` should suffice.
2. **Could "small" be 10× smaller?** If small is mostly a sink, replacing it
   with a much smaller variant should still work and save more compute.

## Background

The criticism: looking at TDLC routing, small dominates at very low SNR where
all bits fail anyway — suggesting small could be replaced by something cheaper
(or removed entirely). Counter-evidence: on UMa, small handles ~40% of traffic
across ALL SNR bins, including high-SNR slots — looks like real decoding work.

## Configs

| Exp | Setup | Tests |
|---|---|---|
| **exp41** | `{nano, large}` — drop small entirely | Is small redundant? |
| **exp42** | Pretrain `dense_micro` (block_dim=16, 8 blocks) | Provides warm-start checkpoint |
| **exp43** | `{nano, micro_small (block_dim=16), large}` MoE | Does smaller "small" suffice? |

All use exp26 recipe otherwise (α=2e-3, asym warm-start, seed 67, 12k steps).

## Cluster

All three: `select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb`
- exp41 (no small): 3h walltime
- exp42 (dense_micro pretrain): 4h walltime
- exp43 (micro-small MoE): 3h walltime — depends on exp42 finishing first

## Results

### exp41 — `{nano, large}` ablation (DONE 2026-04-30)

| Model | UMa BLER | TDLC BLER | Avg BLER |
|---|---:|---:|---:|
| exp26 (3 experts) | 0.937 | 0.867 | **0.902** |
| **exp41 (drop small)** | **0.967** | **0.942** | **~0.955** |

**Dropping small costs +5.3 pp BLER** — much worse than dropping nano (+0.7 pp).
Small is **NOT just a sink** — it does real decoding work, especially on TDLC
where it goes from 0.867 → 0.942 without it.

**Decisively answers the "small could be 10x smaller" critique: no.**

### exp42 + exp43 (in flight)

dense_micro pretrain → then micro-small MoE training. Will tell us whether a
*smaller* small (block_dim 16 instead of 32) can match exp26.

## Implications

- exp41 strongly justifies the 3-expert design with current sizes
- exp43 will tell us whether we can compress the middle expert further
- Either result is publishable
