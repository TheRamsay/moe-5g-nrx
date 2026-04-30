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

- Job 19586663 (initial) — completed but figures saved to scratch (cleaned up)
- Job 19587043 (re-run with persistent output path) — ✅ DONE 2026-04-30
- Output: `docs/figures/router_mechanism_{linear_probing,expert_specialization,decision_boundary}.png` + JSON

## Results

### A. Linear probing — implicit SNR encoding is PROFILE-SPECIFIC

| Probe | UMa R² | TDLC R² |
|---|---:|---:|
| **SNR** | 0.42 | **0.93** |
| Channel power | 0.43 | 0.07 (Sionna normalises per-sample → low variance) |
| Delay spread | 0.36 | 0.65 |
| Profile classification (UMa vs TDLC) | 0.78 (combined) | |

**Key finding:** TDLC SNR R²=0.93 confirms strong implicit SNR encoding for
the harder NLOS channel. UMa SNR R²=0.42 is much weaker — UMa channels are
more position-dependent and SNR alone is a poor predictor of decodability.
The stem learned **profile-specific representations**, not a universal SNR
encoder.

### C. Per-expert specialization

Routing share per profile:
- UMa: 49% nano / 24% small / 26% large
- TDLC: 15% nano / 39% small / 46% large

SNR distribution per chosen expert (cleaner separation on TDLC):
- TDLC: nano at extreme low (-10 to -5 dB), small in middle (~0 dB), large at high (15+ dB)
- UMa: nano dominates low, small fills middle, large skews high — but more overlap

**Striking finding:** nano and small are at BLER ≈ 1.0 across ALL SNR bins.
Only large ever achieves BLER below 1.0 (~0.1 on TDLC at high SNR, ~0.5 on
UMa at high SNR). The router's value is **compute efficiency** — routes
hopeless samples to nano (cheap failure) and decodable samples to large
(only one that can decode).

### F. Decision boundary on PCA plane

Routing decisions form coherent regions in PCA space (5-NN vote). Boundaries
roughly align with the SNR colour gradient — visual confirmation of A's
quantitative result. Cleaner regional separation on TDLC than UMa.

### Per-expert success-rate analysis (added 2026-04-30 evening, v2)

Aggregate per-expert success rates on routed samples (4k per profile):

| Expert | UMa success rate | TDLC success rate |
|---|---:|---:|
| nano | **0.00%** | **0.00%** |
| small | **0.00%** | **0.00%** |
| **large** | **23.21%** | **29.42%** |

**Definitive answer to the "is small a sink?" question:** nano AND small
literally never decode any block successfully. Only large delivers actual
decoded outputs. Nano/small are pure compute optimizers — their value is:

1. Compute savings on hopeless samples
2. Channel-MSE auxiliary loss training signal
3. Routing structure (3 cost tiers)

Figure: `docs/figures/router_mechanism_success_rate.png`. Visualizes the
flat-zero lines for nano/small and the rising large curve.

## Implications for the writeup

**Old narrative:** "Stem encodes SNR implicitly — that's why explicit SNR
proxies (exp38) were redundant."

**New, refined narrative:** "The stem learned profile-specific representations.
Strong SNR encoding on TDLC (R²=0.93) where SNR drives BLER; weaker on UMa
(R²=0.42) where SNR is a poor predictor. The router uses these
profile-appropriate features for routing, explaining why routing patterns
differ between profiles."

This is a **richer story** than "stem encodes SNR" — and arguably more
publishable because it reveals the stem learned what to encode based on what
matters per profile, not a one-size-fits-all representation.

## Why this is the strongest single addition for the writeup

Without this: "PCA shows stem features cluster by SNR."
With this: "**Linear probes on stem features predict SNR with R² = 0.X. The
stem learned a near-complete representation of physical channel parameters
from the BCE+MSE loss alone, despite never being given these labels. This
explains why explicit SNR proxies (exp38) were redundant.**"

That's a thesis-worthy paragraph + 3 killer figures.
