# Compute-Aware 5G Neural Receiver

MoE neural receiver for 5G with compute-aware routing.

## Quick Start

```bash
# Setup (once)
./scripts/metacentrum_setup.sh

# Run locally
uv run python main.py experiment=exp01_baseline training.max_steps=100

# Submit batch job
qsub -v "RUN_ARGS=experiment=exp01_baseline runtime.device=cuda" scripts/metacentrum_job.sh

# Interactive job
./scripts/metacentrum_interactive.sh 2
source scripts/interactive_env.sh
run_experiment 'experiment=exp01_baseline runtime.device=cuda training.batch_size=64'
```

## Local Development

```bash
# Install
uv sync --python 3.10 --dev

# Run
uv run python main.py model=static_dense
uv run python main.py experiment=exp01_baseline dataset=mixed

# Code quality
uv run ruff check . --fix
uv run ruff format .
```

## MetaCentrum HPC

### One-time Setup
```bash
./scripts/metacentrum_setup.sh
```

### Submit Jobs
```bash
# Single job
qsub -v "RUN_ARGS=experiment=exp01_baseline runtime.device=cuda" scripts/metacentrum_job.sh

# With overrides
qsub -v "RUN_ARGS=experiment=exp01_baseline runtime.device=cuda training.batch_size=128" \
     -l walltime=8:00:00 \
     scripts/metacentrum_job.sh

# Check status
qstat -u $USER

# Delete job
qdel <JOBID>
```

### Interactive Jobs
```bash
# Request 2 hours
./scripts/metacentrum_interactive.sh 2

# On compute node
source scripts/interactive_env.sh
run_experiment 'experiment=exp01_baseline runtime.device=cuda'
sync_back
exit
```

## Experiments

```bash
# Copy template
cp -r experiments/2024-03-18-example experiments/2024-03-19-your-batch

# Edit configs in conf/experiment/
# Submit all
bash submit.sh
```

## Hydra Config

Configs live in `conf/`:

```bash
# Switch model
uv run python main.py model=static_dense
uv run python main.py experiment=exp01_baseline

# Override any value
uv run python main.py experiment=exp01_baseline training.batch_size=64
```

Key files:
- `conf/config.yaml` - Main config
- `conf/model/static_dense.yaml` - Dense baseline
- `conf/model/moe.yaml` - MoE model
- `conf/experiment/exp01.yaml` - Experiment variants

## Run Naming

Use three levels consistently in WandB and Hydra presets:

- `project`: whole repo/workstream, keep `moe-5g-nrx`
- `experiment.batch_name`: one study or sweep, e.g. `dense-baseline-v1`
- `experiment.exp_name`: one exact run config, e.g. `dense_s56_b8_h48_bs32_lr1e3_s67`

Recommended tags are broad filters only, for example:

- `baseline`, `dense`, `synthetic`, `simo-1x4`, `16qam`

Example:

```bash
uv run python main.py experiment=exp01_baseline
```

Capacity sweep presets:

```bash
uv run python main.py experiment=exp03_dense_capacity_small
uv run python main.py experiment=exp04_dense_capacity_mid
uv run python main.py experiment=exp05_dense_capacity_large
```

These three runs share the same WandB group `dense-capacity-v1` so they are easy to compare.

## Validation & Test Datasets

Training data is generated dynamically via Sionna. For reproducible evaluation, generate cached validation/test datasets:

```bash
# Generate all datasets (val + test for UMa, TDL-C, mixed)
uv run python scripts/generate_datasets.py

# Generate only validation sets
uv run python scripts/generate_datasets.py generation.split=val

# Custom sample count
uv run python scripts/generate_datasets.py generation.num_samples=16384

# Custom output directory
uv run python scripts/generate_datasets.py generation.output_dir=data2
```

Datasets are saved to `data/val/` and `data/test/` (gitignored).

DeepMIMO OOD datasets are generated separately in Arrow format:

```bash
uv run python scripts/generate_deepmimo_dataset.py \
    generation.split=test \
    generation.num_samples=32768 \
    generation.deepmimo.scenario=asu_campus1 \
    generation.deepmimo.profile_name=deepmimo
```

This produces `data/test/deepmimo/` (directory), logs `dataset-test-deepmimo`,
and can be consumed directly by `scripts/evaluate.py`.
If a scenario has fewer unique channels than requested samples, the generator
automatically resamples channels with replacement to still produce the exact sample count.

### Training Modes

Train on a single channel profile or mixed (alternating UMa/TDL-C):

```bash
# Single profile training
uv run python main.py dataset=tdlc   # TDL-C only
uv run python main.py dataset=uma    # UMa only

# Mixed training (alternates between UMa and TDL-C each batch)
uv run python main.py dataset=mixed
```

For the current dense baselines, use `dataset=mixed` for training and evaluate the trained checkpoint on `uma` and `tdlc` separately.

### Validation During Training

Validation runs periodically during training using cached `uma` and `tdlc` validation datasets:

```bash
# Train with periodic validation (every 500 steps)
uv run python main.py validation.every_n_steps=500

# Validate only on a single cached dataset
uv run python main.py validation.profiles=[] validation.dataset_path=data/val/uma.pt

# Disable validation
uv run python main.py validation.enabled=false
```

Checkpointing during training:

```bash
# Save latest checkpoints every 500 steps and track the best validation checkpoint
uv run python main.py training.checkpoint.every_n_steps=500

# Change checkpoint selection metric
uv run python main.py training.checkpoint.best_metric=ber

# Select best checkpoint by the worst profile instead of the mean
uv run python main.py training.checkpoint.best_metric_aggregation=max
```

By default training now writes:
- `<checkpoint_dir>/<model>.pt` for the final checkpoint
- `<checkpoint_dir>/<model>_latest.pt` for the latest periodic checkpoint
- `<checkpoint_dir>/<model>_best.pt` for the best validation checkpoint

For dense baselines, the default best-checkpoint selection metric is the mean validation `ber` across `uma` and `tdlc`.

### Test Evaluation (Post-Training Only)

Test datasets are used **only after training completes**:

Current canonical dense checkpoint artifact:

- `knn_moe-5g-nrx/moe-5g-nrx/model-dense_large_final20k_constant_lr_s67-55l1dpby:best`

```bash
# Evaluate the canonical dense checkpoint on UMa + TDL-C
uv run python scripts/evaluate.py \
    evaluation.checkpoint_artifact=knn_moe-5g-nrx/moe-5g-nrx/model-dense_large_final20k_constant_lr_s67-55l1dpby:best \
    evaluation.profiles=[uma,tdlc]

# Evaluate OOD DeepMIMO only (Arrow dataset artifact: dataset-test-deepmimo)
uv run python scripts/evaluate.py \
    evaluation.checkpoint_artifact=knn_moe-5g-nrx/moe-5g-nrx/model-dense_large_final20k_constant_lr_s67-55l1dpby:best \
    evaluation.profiles=[deepmimo]

# Evaluate one explicit dataset
uv run python scripts/evaluate.py \
    evaluation.checkpoint_artifact=knn_moe-5g-nrx/moe-5g-nrx/model-dense_large_final20k_constant_lr_s67-55l1dpby:best \
    evaluation.dataset_path=data/test/uma.pt evaluation.profiles=[]

# SNR-binned analysis
uv run python scripts/evaluate.py \
    evaluation.checkpoint_artifact=knn_moe-5g-nrx/moe-5g-nrx/model-dense_large_final20k_constant_lr_s67-55l1dpby:best \
    evaluation.profiles=[uma,tdlc] evaluation.snr_bins=5
```

Evaluation logs a per-profile summary table and scalar metrics to WandB when `logging.use_wandb=true` and `evaluation.log_to_wandb=true`.

When `evaluation.snr_bins` is enabled, eval runs also log:

- `viz/snr_binned_error_heatmap`
- `viz/failure_grid_heatmap`
- `eval/failure_grid`

When possible, evaluation also links back to the checkpoint artifact produced by training and consumes `test/uma` / `test/tdlc` dataset artifacts, so the final eval run has explicit dataset and model lineage in WandB.

## Wandb

```bash
# Login (once)
wandb login

# View results
# https://wandb.ai/your-entity/moe-5g-nrx
```

Env vars:
```bash
export WANDB_PROJECT=my-project
export WANDB_ENTITY=your-username
```

Artifact workflow:

```bash
# Generate cached validation datasets once and reuse them across runs
uv run python scripts/generate_datasets.py generation.split=val generation.num_samples=8192

# Generate cached test datasets once and reuse them across runs
uv run python scripts/generate_datasets.py generation.split=test generation.num_samples=32768

# Generate cached DeepMIMO OOD test dataset (Arrow directory)
uv run python scripts/generate_deepmimo_dataset.py \
    generation.split=test generation.num_samples=32768 \
    generation.deepmimo.scenario=asu_campus1 \
    generation.deepmimo.profile_name=deepmimo

# Train and log the final checkpoint as a model artifact
uv run python main.py experiment=exp01_baseline

# Evaluate a local checkpoint while automatically linking back to its artifact
uv run python scripts/evaluate.py evaluation.checkpoint=checkpoints/static_dense_nrx.pt \
    evaluation.profiles=[uma,tdlc]

# Or evaluate directly from a checkpoint artifact reference
uv run python scripts/evaluate.py \
    evaluation.checkpoint_artifact=your-entity/moe-5g-nrx/model-<exp-name>-<run-id>:latest \
    evaluation.profiles=[uma,tdlc]

# Evaluate ID + OOD in one run
uv run python scripts/evaluate.py \
    evaluation.checkpoint_artifact=your-entity/moe-5g-nrx/model-<exp-name>-<run-id>:latest \
    evaluation.profiles=[uma,tdlc,deepmimo]

# Use the best validation checkpoint artifact instead of the final one
uv run python scripts/evaluate.py \
    evaluation.checkpoint_artifact=your-entity/moe-5g-nrx/model-<exp-name>-<run-id>:best \
    evaluation.profiles=[uma,tdlc]
```

The `manual-smoke` datasets (`1024` samples/profile) are only for plumbing checks. For real dense experiments, regenerate larger cached validation and test sets and reuse them for the whole study.

Report workflow:

```bash
uv run python scripts/create_wandb_report.py logging.entity=your-username
```

This creates a WandB report page with:
- training curves (`train/*`, `val/uma/*`, `val/tdlc/*`)
- final evaluation bar charts
- parameter-count vs BLER scatter plots
- a run comparer for dense baseline/capacity runs

Registry and evaluation tables:

- every train/eval run now records:
  - `registry/run_role`
  - `registry/study_slug`
  - `registry/study_path`
  - `registry/question`
- evaluation runs log structured tables:
  - `eval/comparison` - one row per profile with study/checkpoint/dataset lineage
  - `eval/snr_binned` - one row per SNR bin and profile
  - `eval/failures` - top hard examples ranked by bit errors

These tables are intended to make WandB the experiment registry while keeping artifacts as the source of truth for datasets and checkpoints.

## Results

```
results/
├── <JOBID>/              # Job results
│   ├── wandb/            # Logs
│   └── run.log
└── interactive-*/        # Interactive sessions
```

## Project Structure

```
├── conf/              # Hydra configs
├── experiments/       # Experiment batches
├── scripts/           # HPC scripts
├── src/               # Source code
├── main.py            # Entry point
└── pyproject.toml     # Dependencies
```
