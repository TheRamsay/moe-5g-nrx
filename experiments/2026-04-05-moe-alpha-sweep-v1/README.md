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

## Results (10k steps, beta=0.1 throughout)

| alpha | small | mid | large | entropy | entropy% | val/tdlc loss | val/uma loss | val/tdlc BER | val/uma BER | best score |
|---|---|---|---|---|---|---|---|---|---|---|
| 1e-3 | 33.6% | 33.5% | **33.0%** | 1.098 | **99.9%** | **0.2463** | **0.4416** | **0.1288** | **0.2787** | **0.2034** |
| 1e-2 | 36.7% | 33.9% | **29.3%** | 0.602 | 54.8% | 0.2525 | 0.4478 | 0.1285 | 0.2784 | 0.2034 |
| 5e-3 | 52.5% | 47.5% | **~0** | 0.444 | 40.4% | 0.2562 | 0.4504 | 0.1293 | 0.2785 | 0.2039 |

Artifacts:
- `alpha=1e-3`: `knn_moe-5g-nrx/moe-5g-nrx/model-moe_alpha_v1_a1e_3_b0p1_s67-58fepdqp:best`
- `alpha=1e-2`: `knn_moe-5g-nrx/moe-5g-nrx/model-moe_alpha_v1_a1e_2_b0p1_s67-dhqms5c3:best`
- `alpha=5e-3`: `knn_moe-5g-nrx/moe-5g-nrx/model-moe_alpha_v1_a5e_3_b0p1_s67-ct1pjxbl:best`

## Findings

- **Router collapse is non-linear**: the intermediate α=5e-3 collapsed hardest (large ~0), not the
  highest α. Once α is strong enough to destabilize routing early in training, β=0.1 cannot recover
  it. This suggests a threshold effect rather than a gradual degradation.
- **α=1e-2** causes slow but progressive collapse (large drifted 28.7% → 29.3% over 10k steps —
  stabilizing but well below 33%).
- **α=1e-3** holds near-perfect uniform routing throughout training (entropy 99.9% of max).
- **BER is nearly indistinguishable** across all three runs (~0.129 tdlc, ~0.279 uma) — router
  collapse does not catastrophically hurt accuracy at 10k steps because small+mid are sufficient for
  most inputs.
- **Val loss is the differentiator**: α=1e-3 is measurably better (tdlc loss 0.2463 vs 0.2562).
- **Phase 1 matches the dense large baseline** (joint v0 collapse run: tdlc BER 0.128, uma BER 0.277).
  MoE with healthy routing matches while using all three experts.
- **No meaningful compute savings yet**: at α=1e-3, expected FLOPs ratio is ~70% (soft gating during
  training, all experts active). Real efficiency gains require Phase 2 where the router specializes.

## Conclusion

**α=1e-3 is locked** for Phase 2. Phase 1 confirms MoE can train stably from scratch with balanced
routing and match the dense baseline. The actual BLER vs FLOPs Pareto story requires Phase 2
warm-start experiments where the router learns meaningful expert specialization.
