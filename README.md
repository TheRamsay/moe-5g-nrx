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
sync_back   # Copy results
exit
```

## Experiment Batches

```bash
# Copy template batch
cp -r experiments/2024-03-18-routing-temperature \
     experiments/2024-03-19-your-batch

# Edit and submit
cd experiments/2024-03-19-your-batch
vim experiments.yaml
bash submit.sh
```

## Config

```bash
# Switch model
uv run python main.py model=static_dense
uv run python main.py model=moe

# Override values
uv run python main.py model=moe \
    model.router.temperature=0.5 \
    training.batch_size=64 \
    runtime.device=cuda
```

Key files:
- `conf/config.yaml` - Main config
- `conf/model/static_dense.yaml` - Dense baseline
- `conf/model/moe.yaml` - MoE model
- `conf/wandb/default.yaml` - Logging

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
├── <JOBID>/              # Each job
│   ├── wandb/            # Logs
│   ├── final_checkpoint.pt
│   └── run.log
└── interactive-*/        # Interactive sessions
```

## Project Structure

```
├── conf/              # Hydra configs
├── experiments/       # Experiment batches
├── scripts/           # HPC scripts
├── src/               # Source code (TODO)
├── main.py            # Entry point
└── pyproject.toml     # Dependencies
```

## Requirements

- Python 3.10
- uv
- MetaCentrum account

See `AGENTS.md` for architecture details.
