# MoE Joint v0

## Question

Can a minimal joint MoE match the frozen dense baseline before introducing a compute penalty?

This is the first MoE implementation in the repo.

- shared stem
- channel-aware router on pooled shared features
- 3 heterogeneous experts: `small`, `mid`, `large`
- weighted expert combination during training
- hard top-1 routing during inference

## Reference Dense Baseline

- artifact: `knn_moe-5g-nrx/moe-5g-nrx/model-dense_large_final20k_constant_lr_s67-55l1dpby:best`

## Initial Run

- optimizer: `lr=1e-3`, `wd=1e-4`
- max steps: `10000`
- compute penalty: `alpha=0.0`
- load-balance penalty: `beta=0.0`
- dataset version: `dense-v1`

## Quick Start

Print the command first:

```bash
bash experiments/2026-04-05-moe-joint-v0/submit.sh print
```

Submit on the 16 GB GPU class:

```bash
source experiments/resources/gpu-16gb.sh
DATA_ROOT=$HOME/moe-5g-datasets/dense-v1 \
bash experiments/2026-04-05-moe-joint-v0/submit.sh qsub
```

## Next Step After v0

If the `alpha=0.0` run is stable and competitive, copy this study and sweep a small set of positive compute penalties.
