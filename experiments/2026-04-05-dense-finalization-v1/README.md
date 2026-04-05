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
| `20k_constant_lr` | `0.196518` mean val BER at step `19500` | train run `55l1dpby`, eval run `tdw0ip58`, best checkpoint `knn_moe-5g-nrx/moe-5g-nrx/model-dense_large_final20k_constant_lr_s67-55l1dpby:best` |
| `20k_cosine_decay` | `0.196999` mean val BER at step `20000` | train run `yc1kr036`, eval run `v3xjtdep`, best checkpoint `knn_moe-5g-nrx/moe-5g-nrx/model-dense_large_final20k_cosine_lr_s67-yc1kr036:best` |

## Frozen Dense Baseline

The canonical dense baseline is now frozen as:

- recipe: `exp05_dense_capacity_large`
- optimizer: `lr=1e-3`, `wd=1e-4`
- schedule: none
- max steps: `20000`
- seed: `67`
- checkpoint selection metric: mean validation `ber` across `uma` and `tdlc`
- checkpoint artifact: `knn_moe-5g-nrx/moe-5g-nrx/model-dense_large_final20k_constant_lr_s67-55l1dpby:best`

Cached test metrics for the frozen checkpoint (`tdw0ip58`):

- `uma`: BER `0.268348`, BLER `0.936096`
- `tdlc`: BER `0.122086`, BLER `0.866028`

The cosine run remained competitive, but it lost on the primary dense-baseline selection metric and also trailed the constant-LR run on BER for both profiles on cached test.

## Next Step

Use the frozen checkpoint above as the dense reference for MoE work, then start the minimal joint MoE study.
