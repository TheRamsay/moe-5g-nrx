# Experiment: Dense Baseline v1

**Date:** 2026-03-29  
**Question:** What is the canonical static dense baseline we will compare MoE against?  
**WandB group:** `dense-baseline-v1`

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

## Purpose

This is the single reference dense baseline run for the current project stage.

It is trained on mixed UMa/TDL-C batches and should be evaluated on cached `uma` and `tdlc` test datasets separately.

Use it when you want:

- one clean baseline run
- a default dense checkpoint
- a stable dense reference before or after other sweeps

For model-size comparison, use `experiments/2026-03-29-dense-capacity-v1/` instead.

## Hydra Preset

| Config | WandB run | Notes |
|---|---|---|
| `exp01_baseline` | `dense_s56_b8_h48_bs32_lr1e3_s67` | Canonical dense baseline preset |

## Quick Start

From the repository root:

```bash
DATA_ROOT=$HOME/moe-5g-datasets/dense-v1 \
uv run python main.py experiment=exp01_baseline runtime.device=cuda \
    validation.data_dir=$DATA_ROOT/val \
    training.checkpoint_dir=../artifacts/checkpoints
```

Short smoke run:

```bash
DATA_ROOT=$HOME/moe-5g-datasets/dense-v1 \
uv run python main.py experiment=exp01_baseline runtime.device=cuda training.max_steps=1000 \
    validation.data_dir=$DATA_ROOT/val \
    training.checkpoint_dir=../artifacts/checkpoints
```

Evaluation after training from the best local checkpoint:

```bash
DATA_ROOT=$HOME/moe-5g-datasets/dense-v1 \
uv run python scripts/evaluate.py runtime.device=cuda \
    evaluation.checkpoint=results/<JOBID>/checkpoints/static_dense_nrx_best.pt \
    evaluation.data_dir=$DATA_ROOT/test
```

Evaluation from the best checkpoint artifact:

```bash
uv run python scripts/evaluate.py runtime.device=cuda \
    evaluation.checkpoint_artifact=knn_moe-5g-nrx/moe-5g-nrx/model-<run-id>:best
```

Or from this folder:

```bash
bash submit.sh print
bash submit.sh local
bash submit.sh qsub
```

To override the dataset root or add extra Hydra overrides:

```bash
DATA_ROOT=$HOME/moe-5g-datasets/dense-v1 \
EXTRA_ARGS='training.max_steps=15000 validation.every_n_steps=250' \
bash submit.sh qsub
```

## Current Status

- implemented and training successfully
- mixed-profile training preset is configured
- WandB naming and grouping are configured for training and evaluation
- current metrics: BER, SER, BLER, channel MSE, per-block bit-error summaries
- cached validation during training is implemented
- test evaluation targets `uma` and `tdlc` separately
- best/latest/final checkpointing is enabled
- evaluation can consume the `:best` checkpoint artifact

## Results

| Config | Status | Notes |
|---|---|---|
| `exp01_baseline` | active | record final `run-id`, `job id`, and chosen `:best` checkpoint artifact here |

## Reporting

- WandB report URL: add after first full run
- Selection checkpoint: prefer `model-<run-id>:best` over `:latest`

## Relationship to Capacity Sweep

`exp01_baseline` is the default dense reference run.

The capacity sweep uses three separate presets:

- `exp03_dense_capacity_small`
- `exp04_dense_capacity_mid`
- `exp05_dense_capacity_large`

The `mid` preset is in a similar default capacity region, but it belongs to the
`dense-capacity-v1` group so it can be compared side-by-side with the other sizes.
