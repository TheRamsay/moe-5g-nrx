# MoE Phase 2 — Anti-Collapse Sweep v1

## Scope

Originally submitted as a beta sweep, the study grew to cover four orthogonal anti-collapse
mechanisms under the same study_slug (`2026-04-11-moe-phase2-beta-sweep-v1`):

1. **Beta sweep** — MSE load-balance at higher strengths (exp20, β ∈ {0.5, 1.0, 2.0})
2. **Soft capacity constraint** — penalise any expert exceeding 50% of the batch (exp21)
3. **Switch Transformer auxiliary loss** — n_experts · Σ(fᵢ·Pᵢ) (exp22)
4. **Asymmetric warm-start** — large starts random, nano/small warm (exp23, tracked in a
   sibling study dir: `2026-04-11-moe-phase2-asym-warm-v1`)

## Question

What mechanism recovers routing diversity in warm-started Phase 2 without collapsing BLER?

## Background

Phase 2 v1 (exp17, β=0.1) showed full router collapse: large expert received 100% of traffic
from step 1 and never redistributed. The warm-started large is so much better than nano/small
that the performance gradient overwhelms both the FLOPs penalty and load balance penalty.

Key insight from the sweep so far: **router collapse happens during the frozen phase**,
before experts are trainable. So every candidate mechanism has to work against router probs
directly — it cannot rely on expert specialization catching up.

## Jobs

### Beta sweep (exp20)

| β | exp_name | Job ID | W&B run |
|---|---|---|---|
| 0.5 | moe_phase2_b0p5_s67 | 18935230 | `hv4hz8zy` |
| 1.0 | moe_phase2_b1p0_s67 | 18935231 | `4pafgmm4` |
| 2.0 | moe_phase2_b2p0_s67 | 18935232 | `4r53qiqc` |

### Capacity constraint (exp21)

| weight | factor | exp_name | Job ID | W&B run |
|---|---|---|---|---|
| 0.5 | 1.5 | moe_phase2_cap1p5_w0p5_s67 | 18937002 | _tbd_ |

### Switch aux loss (exp22)

| weight | exp_name | Job ID | W&B run |
|---|---|---|---|
| 0.01 | moe_phase2_switch_aux_w0p01_s67 | 18937003 | _tbd_ |

## What To Watch

- `train/ema/router_entropy` — want >0.3 after unfreezing (step 2000+)
- `train/ema/expert_usage/{nano,small,large}` — want large <80%, meaningful nano/small share
- `val/tdlc/snr_bin_4/bler` @ SNR=17 — should stay ≤0.3 for the run to be worth continuing
- `val/tdlc/bler` overall — should not degrade more than ~0.05 above dense large (0.866)

## Mid-Run Observations (beta sweep @ step ~585)

Snapshot after the first validation checkpoint, still inside the frozen phase:

| β | router_entropy | large | nano | small | val tdlc BLER @ SNR=17 |
|---|---|---|---|---|---|
| 0.5 | 1.9e-6 (dead) | 100.0% | ~0 | ~0 | 0.292 |
| 1.0 | 9.0e-7 (dead) | 100.0% | ~0 | ~0 | 0.286 |
| **2.0** | **0.64 (alive)** | **52%** | **48%** | 0% | **0.581** |

**Key findings:**

- **β=0.5 and β=1.0 collapse just like β=0.1.** MSE load-balance does not degrade
  gracefully — at weak strengths it has no effect at all, even a 10× increase over the
  original Phase 2 value leaves the router locked on large within ~100 steps.
- **β=2.0 holds a ~52/48 nano↔large split** and a healthy 0.64 entropy. First data point
  where MSE load-balance actually prevents collapse.
- **Cost:** b2p0's BLER at SNR=17 is 0.58 (vs. ~0.29 for the collapsed runs), because
  ~half the batch flows through nano. Expected tradeoff.
- **small is dead in all three** — nano and large are the only competitors the router
  considers. Suggests the nano→large gap is the only one wide enough to make the router
  hesitate; small is always dominated by large on the problems it sees.

Critical question for the rest of the run: **does b2p0 retain heterogeneous routing after
unfreezing at step 2000 and can experts specialise to close the BLER gap?**

## Results (final)

| Run | Router entropy | large | nano | small | FLOPs ratio | val tdlc BLER @ SNR=17 | Decision |
|---|---|---|---|---|---|---|---|
| β=0.5 | — | — | — | — | — | — | — |
| β=1.0 | — | — | — | — | — | — | — |
| β=2.0 | — | — | — | — | — | — | — |
| capacity (wf=1.5,w=0.5) | — | — | — | — | — | — | — |
| switch_aux (w=0.01) | — | — | — | — | — | — | — |

(See `2026-04-11-moe-phase2-asym-warm-v1/` for the asymmetric warm-start arm.)

## Decision Criteria

- **Heterogeneous routing + BLER @ SNR=17 < 0.35 (val tdlc)** → promising, run to 12k
- Heterogeneous routing but BLER too high → try weaker regularization, or longer joint phase
  so experts can specialise
- Collapse → mechanism insufficient, rule it out
