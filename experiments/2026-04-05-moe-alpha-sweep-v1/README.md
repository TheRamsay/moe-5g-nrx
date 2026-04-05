# MoE Alpha Sweep v1

## Question

With `beta=0.1` preventing router collapse, how much compute penalty (alpha) can we apply before quality degrades?

## Setup

- base: same architecture as v0 (shared stem + 3 heterogeneous experts)
- locked: `beta=0.1` (load-balance penalty)
- sweep: `alpha ∈ {1e-3, 5e-3, 1e-2}`
- max steps: `10000`
- optimizer: `lr=1e-3`, `wd=1e-4`
- dataset: `dense-v1`
- GPU: 46GB class

## What We Expect

- Higher alpha → more routing to small/mid experts → lower FLOPs ratio
- At some alpha the quality (BER/BLER) should start degrading
- The sweet spot gives meaningful compute savings with minimal quality loss
- Per-profile routing should become more differentiated (hard channels → large, easy → small)

## Quick Start

```bash
source experiments/resources/gpu-46gb.sh
DATA_ROOT=$HOME/moe-5g-datasets/dense-v1 \
ALPHA=1e-3 LOAD_BALANCE_BETA=0.1 \
bash experiments/2026-04-05-moe-alpha-sweep-v1/submit.sh qsub
```
