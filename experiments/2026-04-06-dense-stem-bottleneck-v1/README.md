# Dense Stem Bottleneck v1

## Question

Is the shared stem doing most of the heavy lifting, making expert backbone capacity largely
irrelevant? If reducing `state_dim` (the stem output width) hurts quality significantly, the
stem is the bottleneck — the experts are starved of expressive features regardless of their
size.

## Motivation

The Phase 1 SNR-binned evals showed small BER/BLER gaps between small (168k) and large (450k)
experts — especially on UMA. Two competing explanations:

1. **The task is capacity-insensitive** — even a tiny backbone is enough (→ tested by capacity-floor study)
2. **The stem is the bottleneck** — it produces a fixed 56-channel state; larger experts can't
   benefit if the stem already extracts all learnable features

This study tests explanation 2 by fixing the backbone at large settings (`block_hidden_dim=64`,
8 blocks) and aggressively reducing the stem: `state_dim ∈ {32, 16}` vs the baseline 56.

## Design

Backbone is **held fixed** at large settings to isolate stem capacity:
- `block_hidden_dim=64`, `num_cnn_blocks=8`, `readout_hidden_dim=128`

Only `stem_hidden_dims` and `state_dim` are varied.

| Name | `state_dim` | `stem_hidden_dims` | Est. params | Stem FLOPs |
|---|---:|---|---:|---:|
| large (reference) | 56 | [64, 64] | 450k | 285M |
| stem_s32 | 32 | [32, 32] | ~260k | ~93M |
| stem_s16 | 16 | [16, 16] | ~210k | ~25M |

## Training Recipe

- `lr=1e-3`, `wd=1e-4`, constant LR, `seed=67`, `10k` steps
- Training distribution: mixed online Sionna (`uma` + `tdlc`)
- Validation: cached `dense-v1`

## Quick Start

```bash
bash experiments/2026-04-06-dense-stem-bottleneck-v1/submit.sh print
```

```bash
source experiments/resources/gpu-16gb.sh
DATA_ROOT=$HOME/moe-5g-datasets/dense-v1 \
bash experiments/2026-04-06-dense-stem-bottleneck-v1/submit.sh qsub
```

## Results

| Run | val TDLC BER | val UMA BER | best score | Notes |
|---|---|---|---|---|
| stem_s32 | — | — | — | |
| stem_s16 | — | — | — | |

## Interpretation

- If stem_s32 ≈ large: stem is NOT the bottleneck; backbone capacity (or task difficulty)
  is the limiting factor. Reducing the stem in MoE is safe.
- If stem_s32 >> large: the stem is critical; reducing it hurts. This means the current
  `state_dim=56` stem is doing real work, and the experts need richer features to differentiate.
  For MoE, a wider stem with smaller heterogeneous experts may be the right direction.
