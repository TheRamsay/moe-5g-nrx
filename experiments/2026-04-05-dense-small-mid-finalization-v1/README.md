# Dense Small & Mid Finalization v1

## Motivation

Phase 2 MoE warm-start requires all three expert sizes (small, mid, large) to start from comparable
quality. The canonical dense large checkpoint was trained to 20k steps with tuned hyperparameters
(`lr=1e-3`, `wd=1e-4`). The small and mid checkpoints from the capacity sweep were only trained to
10k steps with the same generic hyperparameters.

Using undertrained small/mid checkpoints in Phase 2 would create a quality asymmetry that biases the
router toward large from day one — defeating the purpose of warm-starting.

This study produces finalized small and mid dense checkpoints on the same recipe as the frozen large
baseline, so all three experts enter Phase 2 on equal footing.

## Setup

- Same recipe as `dense-finalization-v1` (the frozen large baseline)
- `lr=1e-3`, `wd=1e-4`, no scheduler (constant LR)
- `max_steps=20000`
- `seed=67`
- dataset: `dense-v1`
- GPU: 16GB class

## Reference

- Frozen large baseline: `knn_moe-5g-nrx/moe-5g-nrx/model-dense_large_final20k_constant_lr_s67-55l1dpby:best`
  - val/tdlc BER: 0.1221, val/uma BER: 0.2683

## Quick Start

```bash
source experiments/resources/gpu-16gb.sh
DATA_ROOT=$HOME/moe-5g-datasets/dense-v1 \
bash experiments/2026-04-05-dense-small-mid-finalization-v1/submit.sh qsub
```

## Results

| Model | Params | val/tdlc BER | val/uma BER | best score | artifact |
|---|---:|---:|---:|---:|---|
| small | 168,324 | | | | |
| mid | 305,924 | | | | |

*(fill in after runs complete)*
