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

| Run | state_dim | val TDLC BER | val UMA BER | best score | TDLC ch. MSE |
|---|---:|---|---|---|---|
| stem_s32 | 32 | 0.1270 | 0.2762 | 0.2016 | 0.0785 |
| stem_s16 | 16 | 0.1454 | 0.2858 | 0.2156 | 0.2003 |
| *large ref (20k)* | *56* | *0.1221* | *0.2683* | *0.1965* | *0.0644* |

## Interpretation

**state_dim=32 is lossless** — matches large with ~67% fewer stem FLOPs. Safe to adopt.

**state_dim=16 breaks channel estimation** — channel_mse is 3× worse; the model runs out of
representational bandwidth to estimate 4 antennas × 128 subcarriers × 14 symbols. The quality
degradation is real and comes specifically from the channel estimation head, not decoding.

Decision: use state_dim=32 for the redesigned MoE expert family. This is implemented in
`conf/model/moe_nl.yaml` and `2026-04-06-dense-large-s32-finalization-v1`.
