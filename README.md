# Compute-Aware 5G Neural Receiver

Minimal starter repo for a compute-aware neural receiver with two model
families:

- `static_dense`: dense neural receiver baseline
- `moe`: compute-aware mixture-of-experts receiver

Configs are managed with Hydra so teammates can switch model families and
override experiment settings from the command line.

## Prerequisites

- Python `3.10` is required for the TensorFlow 2.x + NVIDIA Sionna stack
- `uv` for environment management and command execution

## Setup

Install Python `3.10` if it is not already available:

```bash
uv python install 3.10
```

Create or sync the environment:

```bash
uv sync --python 3.10
```

Install dev tools too:

```bash
uv sync --python 3.10 --dev
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

## Run

Run the default config:

```bash
uv run python main.py
```

Run the static dense baseline explicitly:

```bash
uv run python main.py model=static_dense
```

Run the MoE variant:

```bash
uv run python main.py model=moe
```

Override config values from the CLI:

```bash
uv run python main.py model=moe training.batch_size=64 runtime.device=cuda
```

Hydra config files live in:

- `conf/config.yaml`
- `conf/model/static_dense.yaml`
- `conf/model/moe.yaml`

## Current Experiment Shape

Shared settings live in `conf/config.yaml`:

- project metadata
- runtime settings
- data shape assumptions
- training defaults
- logging defaults

Model-specific settings live in `conf/model/`:

- `static_dense.yaml` for the dense NRX baseline
- `moe.yaml` for the routed MoE receiver

The current entrypoint prints the fully resolved Hydra config, which makes
it easy to verify overrides before wiring in training or evaluation code.

## Quality Checks

Run Ruff linting:

```bash
uv run ruff check .
```

Auto-fix simple Ruff issues:

```bash
uv run ruff check --fix .
```

Run formatting:

```bash
uv run ruff format .
```

There is no committed test suite yet.

## Notes

- The repo is pinned to Python `3.10`, TensorFlow `2.15.0`,
  `tf-keras==2.15.0`, and `sionna==0.19.2`
- On Apple Silicon macOS, `uv` resolves `tensorflow-macos==2.15.0`
  so `import tensorflow` works correctly
- On this macOS machine, `import sionna` still needs the Mitsuba / LLVM
  runtime configured (`DRJIT_LIBLLVM_PATH`) before ray-tracing features work
- `main.py` sets `TF_USE_LEGACY_KERAS=1` before TensorFlow-family imports
- Hydra output directories are disabled for now, so local runs stay clean
## Experiment Tracking with Weights & Biases

Wandb is configured for standard online mode. Just login once and all runs sync automatically.

### Setup

```bash
# Login to wandb once
wandb login

# Or set API key
export WANDB_API_KEY=your-key-here
```

### Configuration

```bash
# Disable wandb
uv run python main.py wandb=disabled

# Enable wandb (default)
uv run python main.py wandb=default

# Set your wandb entity (username or team)
uv run python main.py logging.entity=your-username
```

### MetaCentrum Workflow

**1. Run training:**
```bash
# Batch job
qsub scripts/metacentrum_job.sh

# Interactive job
source scripts/interactive_env.sh
run_experiment 'model=moe'
exit
```

**2. View results live:**
- Go to https://wandb.ai/your-entity/moe-5g-nrx
- Watch metrics update in real-time as the job runs

### Wandb Environment Variables

Set these to customize behavior:

```bash
export WANDB_PROJECT=my-custom-project
export WANDB_ENTITY=your-username
export WANDB_TAGS="experiment-1,5g"
export WANDB_NOTES="Testing MoE with 3 experts"
```

### What Gets Logged

- Config: Full Hydra config with all hyperparameters
- Metrics: Loss, learning rate, system metrics (GPU/CPU/memory)
- Code: Source code snapshot (optional)
- Hardware: PBS job ID, hostname, CUDA devices
- Artifacts: Model checkpoints (if `logging.log_model=true`)
- Artifacts: Model checkpoints (if `logging.log_model=true`)
