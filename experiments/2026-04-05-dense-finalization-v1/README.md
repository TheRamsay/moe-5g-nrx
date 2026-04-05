# Dense Finalization v1

## Question

Can the current best dense recipe improve further with longer training, and does cosine LR decay help late-stage convergence versus a constant learning rate?

This study is intentionally small and targeted.

- base experiment: `exp05_dense_capacity_large`
- optimizer point: `lr=1e-3`, `wd=1e-4`
- seed: `67`
- training distribution: mixed online Sionna (`uma` + `tdlc`)
- validation/test datasets: cached `dense-v1`

## Compared Runs

1. `20k` steps, constant LR
2. `20k` steps, cosine decay to `1e-5`

## Quick Start

Print the commands first:

```bash
bash experiments/2026-04-05-dense-finalization-v1/submit.sh print
```

Submit both runs on the 16 GB GPU class:

```bash
source experiments/resources/gpu-16gb.sh
DATA_ROOT=$HOME/moe-5g-datasets/dense-v1 \
bash experiments/2026-04-05-dense-finalization-v1/submit.sh qsub
```

## Fixed Overrides

- `training.learning_rate=1e-3`
- `training.weight_decay=1e-4`
- `training.max_steps=20000`
- `validation.every_n_steps=500`
- `logging.log_every_n_steps=50`
- `training.checkpoint_dir=../artifacts/checkpoints` in cluster mode

## Results

| Run | Best validation metric seen | Notes |
|---|---|---|
| `20k_constant_lr` | pending | pending |
| `20k_cosine_decay` | pending | pending |

## Next Step

Freeze the winning dense checkpoint and then start the minimal joint MoE study.
