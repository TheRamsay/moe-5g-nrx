# MoE with Explicit SNR-Proxy Router Input — exp38 (v1)

## Question

Does feeding the router **explicit signal-derived statistics** (SNR proxies)
beyond the implicit pooled stem features improve routing — closing the gap
to the SNR-oracle cascade (49% FLOPs vs our 56%)?

## Background

Oracle cascade analysis showed a 7-pp FLOPs gap between exp26 (no SNR access)
and a hand-rule cascade with true SNR. Hypothesis: explicit channel statistics
(received power, channel power, channel variance — all computable at inference
from the existing inputs) might give the router enough information to close
that gap. The model already has the `use_input_statistics` flag in
`src/models/moe.py`; this is the first run that turns it on.

## Configs

| Exp | Recipe | Routing input |
|---|---|---|
| **exp38** | exp26 base (α=2e-3, asym warm-start, seed 67) | pooled stem + (received_power, channel_power, channel_variance) |
| Reference: exp26 | identical | pooled stem only (implicit) |

## Cluster

- Resources: `select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb`
- Walltime: 3h (12k steps ≈ 40 min on GPU)

## Result

**Negative result — collapsed to 100% large.**

| Model | UMa BLER | TDLC BLER | Avg BLER | FLOPs % | Routing |
|---|---:|---:|---:|---:|---|
| exp26 (implicit) | 0.937 | 0.867 | 0.902 | 56% | 44/15/40 |
| **exp38 (explicit SNR proxies)** | 0.935 | 0.859 | 0.897 | **100%** | **100/0/0** |

Adding raw signal statistics caused immediate Phase-2-style collapse to large.
The router uses the explicit signal as an "easy classifier" for picking large
from step 1.

## Implications

**Positive interpretation:** the implicit stem features are *sufficient* — they
already encode SNR-correlated information (confirmed later by PCA visualisation
of stem features colored by true SNR). Adding raw statistics on top is
redundant and destabilising.

**Connects three findings:**
1. Random-router ablation (exp30): channel features matter
2. exp38 collapse: raw stats are redundant with stem features
3. PCA viz: stem features encode SNR implicitly

**For the report:** explicit-SNR-input is NOT future work; we tried it. The
remaining gap to oracle would require a properly trained SNR-estimator module,
not raw signal statistics.
