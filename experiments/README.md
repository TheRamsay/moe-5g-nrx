# Experiments Directory

Use this folder for experiment logs and batch-level orchestration.

- `conf/experiment/*.yaml` stores the actual Hydra run presets.
- `experiments/<date>-<study>/` stores the human-facing study log:
    - hypothesis
    - which configs belong to the study
    - how to launch them
    - results and decisions

This split keeps runnable configs in one place and makes each real sweep reproducible.

## Recommended Workflow

1. Debug with direct CLI overrides.
2. Once a run is worth keeping, create a preset in `conf/experiment/`.
3. Once several presets belong to one scientific question, create a dated folder in `experiments/`.
4. Document the study and launch the batch from that folder.

## Creating a New Experiment Batch

```bash
# 1. Create a dated study folder
mkdir experiments/2026-03-29-your-study

# 2. Copy the template submit script
cp experiments/template.sh experiments/2026-03-29-your-study/submit.sh

# 3. Add or reuse Hydra presets in conf/experiment/
#    e.g. conf/experiment/exp03_dense_capacity_small.yaml

# 4. Document the study in experiments/2026-03-29-your-study/README.md

# 5. Launch
bash experiments/2026-03-29-your-study/submit.sh
```

## Structure

```text
experiments/
├── README.md
├── template.sh
├── 2026-03-29-dense-baseline-v1/
│   ├── README.md
│   └── submit.sh
└── 2026-03-29-dense-capacity-v1/
    ├── README.md
    └── submit.sh

conf/experiment/
├── exp01_baseline.yaml
├── exp02_moe.yaml
├── exp03_dense_capacity_small.yaml
├── exp04_dense_capacity_mid.yaml
└── exp05_dense_capacity_large.yaml
```

## What to Commit

Commit:

- paper-related sweeps
- baselines used in reports
- final ablations you may cite later

Skip committing:

- one-off debug runs
- dead-end tuning noise
- local scratch notes that do not affect conclusions

## Guidelines

1. One hypothesis per batch.
2. Keep the README up to date as results come in.
3. Use one WandB group per study.
4. Record the selection criterion before comparing runs.
5. Prefer descriptive dated folder names over generic `exp01`-style labels.

## Current Batches

- `2026-03-29-dense-baseline-v1/` - canonical single-run dense baseline reference
- `2026-03-29-dense-capacity-v1/` - dense baseline capacity sweep
