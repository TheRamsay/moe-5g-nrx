# Experiment: Dense Capacity Sweep v1

**Date:** 2026-03-29  
**Question:** What dense baseline capacity is strong enough without being unnecessarily large?  
**WandB group:** `dense-capacity-v1`

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

We now have a fixed single-user SIMO dense receiver family in `src/models/dense.py`.
Before comparing against MoE, we need to know whether the dense baseline is underpowered, reasonably sized, or oversized.

This study keeps the architecture family fixed and changes only model capacity.

All runs in this sweep should train on mixed UMa/TDL-C batches and be compared on cached `uma` and `tdlc` validation/test datasets separately.

`exp01_baseline` remains the canonical standalone dense baseline preset. The `mid`
run in this sweep lives in the `dense-capacity-v1` group so it can be compared
directly against `small` and `large` within one study.

## Selection Rule

Choose the smallest dense model that is close to the best performer.

- primary selection metric: mean validation `ber` across `uma` and `tdlc`
- tie-breaker: prefer the smaller model if metrics are close
- final testing should evaluate the `:best` checkpoint artifact, not the final checkpoint
- if two models are very close, prefer the smaller one

## Runs in This Study

| Config | WandB run | Params | Notes |
|---|---|---:|---|
| `exp03_dense_capacity_small` | `dense_small_s56_b8_h32_bs32_lr1e3_s67` | ~168k | Small baseline |
| `exp04_dense_capacity_mid` | `dense_mid_s56_b8_h48_bs32_lr1e3_s67` | ~306k | Mid-capacity sweep run |
| `exp05_dense_capacity_large` | `dense_large_s56_b8_h64_bs32_lr1e3_s67` | ~450k | Larger dense baseline |

## What Is Fixed

- model family: static dense CNN
- state depth: `56`
- number of residual blocks: `8`
- batch size: `32`
- learning rate: `1e-3`
- seed: `67`
- training dataset: online Sionna mixed generator (`uma` + `tdlc`)
- validation dataset: cached Sionna-generated `uma` + `tdlc`
- training budget: currently `10k` steps unless overridden

## Quick Start

From the repository root:

```bash
DATA_ROOT=$HOME/moe-5g-datasets/dense-v1 \
uv run python main.py experiment=exp03_dense_capacity_small runtime.device=cuda \
    validation.data_dir=$DATA_ROOT/val \
    training.checkpoint_dir=../artifacts/checkpoints
DATA_ROOT=$HOME/moe-5g-datasets/dense-v1 \
uv run python main.py experiment=exp04_dense_capacity_mid runtime.device=cuda \
    validation.data_dir=$DATA_ROOT/val \
    training.checkpoint_dir=../artifacts/checkpoints
DATA_ROOT=$HOME/moe-5g-datasets/dense-v1 \
uv run python main.py experiment=exp05_dense_capacity_large runtime.device=cuda \
    validation.data_dir=$DATA_ROOT/val \
    training.checkpoint_dir=../artifacts/checkpoints
```

Or from this folder:

```bash
bash submit.sh print
bash submit.sh local
bash submit.sh qsub
```

## Suggested First Pass

Run shorter smoke comparisons before the full `10k`-step sweep:

```bash
DATA_ROOT=$HOME/moe-5g-datasets/dense-v1 \
uv run python main.py experiment=exp03_dense_capacity_small runtime.device=cuda training.max_steps=1000 \
    validation.data_dir=$DATA_ROOT/val training.checkpoint_dir=../artifacts/checkpoints
DATA_ROOT=$HOME/moe-5g-datasets/dense-v1 \
uv run python main.py experiment=exp04_dense_capacity_mid runtime.device=cuda training.max_steps=1000 \
    validation.data_dir=$DATA_ROOT/val training.checkpoint_dir=../artifacts/checkpoints
DATA_ROOT=$HOME/moe-5g-datasets/dense-v1 \
uv run python main.py experiment=exp05_dense_capacity_large runtime.device=cuda training.max_steps=1000 \
    validation.data_dir=$DATA_ROOT/val training.checkpoint_dir=../artifacts/checkpoints
```

Then inspect the WandB group and compare:

- `val/uma/ber`
- `val/tdlc/ber`
- `checkpoint/best_score`
- `train/loss`

Post-training evaluation from the best local checkpoint:

```bash
DATA_ROOT=$HOME/moe-5g-datasets/dense-v1 \
uv run python scripts/evaluate.py runtime.device=cuda \
    evaluation.checkpoint=results/<JOBID>/checkpoints/static_dense_nrx_best.pt \
    evaluation.data_dir=$DATA_ROOT/test
```

Or evaluate directly from the best checkpoint artifact:

```bash
uv run python scripts/evaluate.py runtime.device=cuda \
    evaluation.checkpoint_artifact=knn_moe-5g-nrx/moe-5g-nrx/model-<exp-name>-<run-id>:best
```

## Current Status

- dense model family is implemented and trains successfully
- WandB naming/grouping is now clean
- runs now train on mixed UMa/TDL-C by default via experiment presets
- training metrics include BER, SER, BLER, channel MSE, and per-block bit-error summaries
- cached validation is implemented during training
- evaluation can report and log separate `uma` and `tdlc` results
- best/latest/final checkpointing is enabled

## Results

| Config | Status | Best metric seen | Notes |
|---|---|---|---|
| `exp03_dense_capacity_small` | active | - | record `run-id`, `job id`, and `:best` checkpoint artifact |
| `exp04_dense_capacity_mid` | active | - | record `run-id`, `job id`, and `:best` checkpoint artifact |
| `exp05_dense_capacity_large` | active | - | record `run-id`, `job id`, and `:best` checkpoint artifact |

## Reporting

- WandB report URL: add after the first full sweep
- Preferred test input: `model-<exp-name>-<run-id>:best` evaluated on cached `$DATA_ROOT/test/{uma,tdlc}.pt`

## Next Step After This Study

After selecting the best size region:

1. sweep learning rate on the chosen capacity
2. rerun the best dense setup across multiple seeds
3. freeze the final dense baseline before MoE comparison
4. add OOD evaluation later as a separate study
