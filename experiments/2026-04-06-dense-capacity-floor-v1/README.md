# Dense Capacity Floor v1

## Question

Is there a meaningful capacity floor below the existing "small" expert (`block_hidden_dim=32`)?
The SNR-binned test evals showed surprisingly small BER/BLER differences between small (168k) and
large (450k) — especially on UMA. This study investigates whether the task is genuinely
capacity-insensitive, or whether the current expert width range (32–64) sits above a floor that
matters.

## Motivation

From the Phase 1 eval results:
- UMA high-SNR (22.9 dB): large BLER=0.759, small BLER=0.804 — only ~4.5pp gap
- TDL-C high-SNR (17.9 dB): large BLER=0.192, small BLER=0.416 — 2× gap, more signal

If a nano model (block_hidden_dim=8, 4 blocks, ~90k params) performs comparably to small,
the MoE expert width range is too conservative and Phase 2 will yield a trivial Pareto curve.
If nano degrades significantly, we have a capacity cliff to exploit.

**Design choice:** stem is held fixed at `[64, 64]` (same as large) to isolate backbone capacity.
Only backbone width and depth are varied.

## Variants

| Name | `block_hidden_dim` | `num_cnn_blocks` | `readout_hidden_dim` | Est. params |
|---|---:|---:|---:|---:|
| nano  | 8  | 4 | 32  | ~90k  |
| micro | 16 | 4 | 64  | ~104k |
| small (reference) | 32 | 8 | 96 | 168k |
| large (reference) | 64 | 8 | 128 | 450k |

## Reference Baselines

From `2026-04-05-dense-small-mid-finalization-v1` test evals:

| Model | TDLC BER | TDLC BLER@17.9dB | UMA BER | UMA BLER@22.9dB |
|---|---|---|---|---|
| small | 0.1258 | 0.416 | 0.2717 | 0.804 |
| large | 0.1221 | 0.192 | 0.2683 | 0.759 |

## Training Recipe

- Same as capacity sweep: `lr=1e-3`, `wd=1e-4`, constant LR, `seed=67`, `10k` steps
- Training distribution: mixed online Sionna (`uma` + `tdlc`)
- Validation: cached `dense-v1`

## Quick Start

```bash
bash experiments/2026-04-06-dense-capacity-floor-v1/submit.sh print
```

```bash
source experiments/resources/gpu-16gb.sh
DATA_ROOT=$HOME/moe-5g-datasets/dense-v1 \
bash experiments/2026-04-06-dense-capacity-floor-v1/submit.sh qsub
```

## Results

| Run | Params | val TDLC BER | val UMA BER | best score | val TDLC BLER | val UMA BLER |
|---|---:|---|---|---|---|---|
| nano  | 90k  | 0.1323 | 0.2814 | 0.2064 | 0.9711 | 0.9753 |
| micro | 104k | 0.1296 | 0.2791 | 0.2044 | 0.9492 | 0.9617 |
| *small (20k ref)* | *168k* | *0.1251* | *0.2751* | *0.2001* | *0.9109* | *0.9513* |
| *large (20k ref)* | *450k* | *0.1221* | *0.2683* | *0.1965* | *0.8660* | *0.9361* |

## Interpretation

**BER gap is tiny (~1pp) but BLER gap is real (~7-8pp TDLC)**. The BER metric averages over
low-SNR bins where all models get BLER=1.0 regardless of capacity, masking the real signal.
BLER captures the waterfall transition shift — larger models achieve the same BLER at lower SNR.

Conclusion: capacity does matter, just not where BER looks. The MoE expert range should be
widened (nano/small/large = block_dim 8/32/64) and evaluated on **BLER@high-SNR** rather
than average BER. This is implemented in `2026-04-06-moe-nl-phase1-v1`.
