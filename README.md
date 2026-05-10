# Compute-Aware 5G Neural Receiver

MoE neural receiver for 5G with compute-aware routing.

## Quick Start

```bash
# Install dependencies
uv sync --python 3.10 --dev

# Run a short local smoke test
uv run python main.py experiment=exp01_baseline training.max_steps=100

# Run a configured experiment locally
uv run python main.py experiment=exp26_moe_alphasweep_a2e3
```

## MetaCentrum

```bash
# One-time setup on MetaCentrum
./scripts/metacentrum_setup.sh

# Submit a single Hydra run
qsub -v "RUN_ARGS=experiment=exp01_baseline runtime.device=cuda" scripts/metacentrum_job.sh

# Submit with overrides
qsub -v "RUN_ARGS=experiment=exp01_baseline runtime.device=cuda training.batch_size=128" \
     -l walltime=8:00:00 \
     scripts/metacentrum_job.sh

# Start an interactive allocation
./scripts/metacentrum_interactive.sh 2
source scripts/interactive_env.sh
run_experiment 'experiment=exp01_baseline runtime.device=cuda'
sync_back
```

The top-level `scripts/` directory contains shell helpers used for setup,
interactive sessions, batch jobs, and dataset predownload/export jobs.

## Experiments

Experiment batches live in `experiments/`. Most batch directories contain one
or more `submit*.sh` scripts with the exact commands used for the study.

```bash
# Example: submit an existing experiment batch
cd experiments/2026-05-01-synthesis-sink-small-large-v1
bash submit.sh
```

Hydra presets live in `conf/experiment/` and can also be launched directly:

```bash
uv run python main.py experiment=exp03_dense_capacity_small
uv run python main.py experiment=exp04_dense_capacity_mid
uv run python main.py experiment=exp05_dense_capacity_large
```

## Configuration

Configs live in `conf/`:

- `conf/config.yaml` - main Hydra config
- `conf/dataset/` - channel and dataset presets
- `conf/model/static_dense.yaml` - dense baseline
- `conf/model/moe.yaml` - standard MoE model
- `conf/model/moe_sink_small_large.yaml` - final sink/small/large MoE variant
- `conf/experiment/` - named experiment and evaluation presets

Common overrides:

```bash
uv run python main.py model=static_dense
uv run python main.py dataset=mixed
uv run python main.py experiment=exp01_baseline training.batch_size=64
uv run python main.py validation.enabled=false
```

## Outputs

Training writes checkpoints under the configured checkpoint directory:

- `<checkpoint_dir>/<model>.pt` - final checkpoint
- `<checkpoint_dir>/<model>_latest.pt` - latest periodic checkpoint
- `<checkpoint_dir>/<model>_best.pt` - best validation checkpoint

WandB logging is controlled by `conf/wandb/` and the standard environment
variables:

```bash
export WANDB_PROJECT=moe-5g-nrx
export WANDB_ENTITY=<your-entity>
wandb login
```

## Project Structure

```text
├── conf/              # Hydra configs
├── experiments/       # Experiment batches and job scripts
├── report/            # LaTeX report sources and figures
├── scripts/           # MetaCentrum shell helpers
├── src/               # Source code
├── main.py            # Training entry point
├── projekt.pdf        # Final compiled report
├── pyproject.toml     # Project metadata and dependencies
└── uv.lock            # Locked Python environment
```
