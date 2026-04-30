# Router Mechanism Analysis (v1)

## Question

What does the router actually *do*, mechanistically? Three concrete sub-questions:

1. **A. Linear probing:** what physical channel parameters do the stem features encode?
2. **C. Per-expert specialization:** which samples does each expert see, and what BLER does it deliver per SNR bin?
3. **F. Decision boundary:** where in feature space does the router switch experts, and does it align with SNR?

## Background

We already have:
- PCA visualization showing stem features cluster by SNR (qualitative)
- Routing trajectories showing when each paradigm commits (training dynamics)
- Random-router ablation showing channel features are causal (counterfactual)

**Missing: quantitative + per-sample analysis** — what makes the router pick
each expert, what each expert delivers, where the decision boundary sits.

This experiment closes those gaps with one inference pass over UMa + TDLC
test data (4k samples each).

## Configs

`scripts/analyze_router_mechanism.py` — single script, three analyses:

| Analysis | Output |
|---|---|
| **A. Linear probing** | Train tiny linear regressors on pooled stem features (112-dim) to predict true SNR / channel power / delay spread. Train logistic-style classifier for profile (UMa vs TDLC). 70/30 train/test split, R² + accuracy reported. |
| **C. Per-expert specialization** | Histogram of SNR per chosen expert; BLER per expert across SNR bins; routing share bar chart per profile. |
| **F. Decision boundary** | 2D PCA of stem features, then 5-NN vote on a dense grid in PCA space to color routing decision regions. Overlay actual samples colored by true SNR. |

All three use the same loaded checkpoint + data, single GPU pass.

## Cluster

- Resources: `select=1:ncpus=4:ngpus=1:mem=16gb:scratch_ssd=10gb`
- Walltime: 1h (most time spent on PTX JIT + W&B artifact download)

## Predicted outcomes

### A — Linear probing
- **profile_acc**: ~0.99 (UMa vs TDLC is easy)
- **uma/snr_r2, tdlc/snr_r2**: ~0.85-0.95 (high — stem encodes SNR well)
- **uma/chan_power_r2, tdlc/chan_power_r2**: high (>0.7) — channel power directly observable from input
- **uma/delay_spread_r2, tdlc/delay_spread_r2**: moderate (0.4-0.7) — harder feature

If SNR R² > 0.9 → "the stem learned the physics from the BCE+MSE loss alone."
This is a **publication-worthy quantitative claim**.

### C — Per-expert specialization
- nano routed samples: concentrated at low SNR (UMa) and very low SNR (TDLC)
- small routed samples: mid-low SNR for both profiles
- large routed samples: waterfall + high SNR

Expect to see "the router picks experts where they actually deliver value."

### F — Decision boundary
- The 5-NN vote should produce contiguous regions per expert
- Expert regions should align approximately with SNR gradient
- "Boundaries are visually monotonic in SNR" → confirms the hypothesis from A

## Status

Job 19586663 submitted 2026-04-30 evening. Output figures + JSON in
`docs/figures/router_mechanism_*`.

## Why this is the strongest single addition for the writeup

Without this: "PCA shows stem features cluster by SNR."
With this: "**Linear probes on stem features predict SNR with R² = 0.X. The
stem learned a near-complete representation of physical channel parameters
from the BCE+MSE loss alone, despite never being given these labels. This
explains why explicit SNR proxies (exp38) were redundant.**"

That's a thesis-worthy paragraph + 3 killer figures.
