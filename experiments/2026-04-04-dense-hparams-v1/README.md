# Experiment: Dense Hyperparameter Sweep v1

**Date:** 2026-04-04  
**Question:** Which optimizer settings work best for the selected dense capacity?  
**WandB group:** `dense-hparams-v1`

**When to use this study:** after `dense-capacity-v1` identifies the winning dense capacity.

**Dataset root:** set via `DATA_ROOT`, expected layout:

```text
$DATA_ROOT/
├── val/
│   ├── uma.pt
│   └── tdlc.pt
└── test/
    ├── uma.pt
    └── tdlc.pt
```

## Motivation

The capacity sweep should answer the architecture question first: small, mid, or large.
This study answers the next question: given the chosen dense capacity, does a nearby optimizer configuration improve validation BER enough to matter?

This batch deliberately tunes only a small set of optimizer knobs:

- `training.learning_rate`
- `training.weight_decay`

It keeps everything else fixed so the comparison stays interpretable.

## Sweep Matrix

Default grid:

- learning rate: `3e-4`, `1e-3`, `3e-3`
- weight decay: `0`, `1e-4`

That gives `6` runs total.

The chosen base experiment is supplied through `BASE_EXPERIMENT`.
Examples:

- `exp03_dense_capacity_small`
- `exp04_dense_capacity_mid`
- `exp05_dense_capacity_large`

## Selection Rule

- primary metric: mean validation `ber` across `uma` and `tdlc`
- tie-breaker: prefer the simpler optimizer setting if performance is effectively equal
- final test evaluation should use the `:best` checkpoint artifact of the winning hyperparameter run

## Quick Start

Print the commands first:

```bash
BASE_EXPERIMENT=exp04_dense_capacity_mid \
bash experiments/2026-04-04-dense-hparams-v1/submit.sh print
```

Submit the full sweep:

```bash
BASE_EXPERIMENT=exp04_dense_capacity_mid \
DATA_ROOT=$HOME/moe-5g-datasets/dense-v1 \
bash experiments/2026-04-04-dense-hparams-v1/submit.sh qsub
```

Adjust the grid if needed:

```bash
BASE_EXPERIMENT=exp04_dense_capacity_mid \
LEARNING_RATES='5e-4 1e-3 2e-3' \
WEIGHT_DECAYS='0 1e-5 1e-4' \
MAX_STEPS=12000 \
VALIDATION_EVERY_N_STEPS=250 \
bash experiments/2026-04-04-dense-hparams-v1/submit.sh qsub
```

## What Is Fixed

- base dense architecture is inherited from `BASE_EXPERIMENT`
- training dataset: online Sionna mixed generator (`uma` + `tdlc`)
- validation dataset: cached Sionna-generated `uma` + `tdlc`
- seed: `67`
- best/latest/final checkpointing remains enabled

## What Changes Per Run

- `training.learning_rate`
- `training.weight_decay`
- `experiment.exp_name`

## Results

| Base experiment | Run | Best metric seen | Notes |
|---|---|---|---|
| `<fill after capacity winner>` | `lr3e-4_wd0` | - | |
| `<fill after capacity winner>` | `lr3e-4_wd1e-4` | - | |
| `<fill after capacity winner>` | `lr1e-3_wd0` | - | |
| `<fill after capacity winner>` | `lr1e-3_wd1e-4` | - | |
| `<fill after capacity winner>` | `lr3e-3_wd0` | - | |
| `<fill after capacity winner>` | `lr3e-3_wd1e-4` | - | |

## Reporting

- WandB report URL: add after the sweep finishes
- Preferred test input: the winning run's `model-<exp-name>-<run-id>:best`

## Next Step After This Study

1. rerun the best dense setup across multiple seeds
2. evaluate the best checkpoint on cached `uma` and `tdlc` test sets
3. freeze the final dense baseline before MoE comparison
