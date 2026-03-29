# Experiment: Dense Baseline v1

**Date:** 2026-03-29  
**Question:** What is the canonical static dense baseline we will compare MoE against?  
**WandB group:** `dense-baseline-v1`

## Purpose

This is the single reference dense baseline run for the current project stage.

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
uv run python main.py experiment=exp01_baseline
```

Short smoke run:

```bash
uv run python main.py experiment=exp01_baseline training.max_steps=1000
```

Or from this folder:

```bash
bash submit.sh print
bash submit.sh local
```

## Current Status

- implemented and training successfully
- WandB naming and grouping are configured
- current metrics: BER, SER, BLER, per-block bit-error summaries
- validation is still pending

## Results

| Config | Status | Notes |
|---|---|---|
| `exp01_baseline` | pending | |

## Relationship to Capacity Sweep

`exp01_baseline` is the default dense reference run.

The capacity sweep uses three separate presets:

- `exp03_dense_capacity_small`
- `exp04_dense_capacity_mid`
- `exp05_dense_capacity_large`

The `mid` preset is in a similar default capacity region, but it belongs to the
`dense-capacity-v1` group so it can be compared side-by-side with the other sizes.
