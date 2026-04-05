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

## Initial Run (alpha=0, beta=0, 10k steps)

- optimizer: `lr=1e-3`, `wd=1e-4`
- max steps: `10000`
- compute penalty: `alpha=0.0`
- load-balance penalty: `beta=0.0`
- dataset version: `dense-v1`
- artifact: `knn_moe-5g-nrx/moe-5g-nrx/model-moe_joint_v0_a0p0_b0p0_s67-hennglkp:best`

### Result: Router Collapse

Router collapsed completely to large expert within early steps:

| Metric | Value |
|---|---|
| expert_usage/large | 1.000 |
| expert_usage/mid | ~0 |
| expert_usage/small | ~0 |
| router_entropy | ~0 |
| val/tdlc BER | 0.128 |
| val/uma BER | 0.277 |

Without load-balance penalty, no incentive to spread load. Model degenerates to single-expert dense equivalent.

## Alpha-Beta Sweep (2k steps)

Swept `alpha ∈ {0, 1e-5, 1e-4}` × `beta ∈ {0.01, 0.1}` for 2k steps to find working regularization.

| alpha | beta | Entropy | Large | Mid | Small | FLOPs ratio | val/tdlc BER | val/uma BER |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.01 | 0.444 | 0.483 | 0.224 | 0.293 | 0.762 | 0.137 | 0.284 |
| 0 | 0.1 | 0.883 | 0.349 | 0.349 | 0.302 | 0.717 | 0.136 | 0.284 |
| 1e-5 | 0.01 | 0.335 | 0.546 | 0.265 | 0.189 | 0.808 | 0.135 | 0.282 |
| 1e-5 | 0.1 | 0.905 | 0.352 | 0.353 | 0.296 | 0.719 | 0.136 | 0.284 |
| 1e-4 | 0.01 | 0.385 | 0.508 | 0.273 | 0.219 | 0.788 | 0.135 | 0.283 |
| 1e-4 | 0.1 | 0.883 | 0.358 | 0.341 | 0.301 | 0.720 | 0.136 | 0.284 |

### Findings

- `beta=0.1` prevents collapse (entropy ~0.88-0.91); `beta=0.01` is too weak.
- `alpha` at 1e-5 and 1e-4 has negligible effect — need larger values.
- Quality is comparable across all runs at 2k steps.
- Per-profile routing already shows domain-aware behavior:
  - TDLC (harder): prefers large/mid experts (~39/38/22%)
  - UMA: prefers small expert (~31/30/39%)

## Conclusion

`beta=0.1` is the minimum viable load-balance penalty. Alpha needs to be much larger (1e-3+) to actually drive compute savings. Proceed to v1 alpha sweep.
