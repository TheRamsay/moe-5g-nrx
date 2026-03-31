# Compute-Aware 5G Neural Receiver

MoE neural receiver for 5G with compute-aware routing.

## Quick Start

```bash
# Setup (once)
./scripts/metacentrum_setup.sh

# Run locally
uv run python main.py model=moe training.max_steps=100

# Submit batch job
qsub -v "RUN_ARGS=model=moe" scripts/metacentrum_job.sh

# Interactive job
./scripts/metacentrum_interactive.sh 2
source scripts/interactive_env.sh
run_experiment 'model=moe training.batch_size=64'
```

## Local Development

```bash
# Install
uv sync --python 3.10 --dev

# Run
uv run python main.py model=static_dense
uv run python main.py model=moe training.batch_size=64 runtime.device=cuda

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
qsub -v "RUN_ARGS=model=moe" scripts/metacentrum_job.sh

# With overrides
qsub -v "RUN_ARGS=model=moe training.batch_size=128" \
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
run_experiment 'model=moe'
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
uv run python main.py model=moe

# Override any value
uv run python main.py model=moe training.batch_size=64
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
uv run python scripts/generate_datasets.py --split val

# Custom sample count
uv run python scripts/generate_datasets.py --num-samples 16384
```

Datasets are saved to `data/val/` and `data/test/` (gitignored).

### Training Modes

Train on a single channel profile or mixed (alternating UMa/TDL-C):

```bash
# Single profile training
uv run python main.py dataset=tdlc   # TDL-C only
uv run python main.py dataset=uma    # UMa only

# Mixed training (alternates between UMa and TDL-C each batch)
uv run python main.py dataset=mixed
```

### Validation During Training

Validation runs periodically during training using cached val datasets:

```bash
# Train with periodic validation (every 500 steps)
uv run python main.py validation.every_n_steps=500

# Disable validation
uv run python main.py validation.enabled=false
```

### Test Evaluation (Post-Training Only)

Test datasets are used **only after training completes**:

```bash
# Evaluate on default test set
uv run python scripts/evaluate.py --checkpoint checkpoints/static_dense.pt

# Evaluate on all channel profiles
uv run python scripts/evaluate.py --checkpoint checkpoints/model.pt --all-profiles

# SNR-binned analysis
uv run python scripts/evaluate.py --checkpoint checkpoints/model.pt --snr-bins 5
```

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
