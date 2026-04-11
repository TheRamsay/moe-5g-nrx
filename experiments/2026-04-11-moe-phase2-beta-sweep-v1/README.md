# MoE Phase 2 Beta Sweep v1

## Question

What load_balance_beta recovers routing diversity in warm-started Phase 2 without collapsing BLER?

## Background

Phase 2 v1 (beta=0.1) showed full router collapse: large expert received 100% of traffic from
step 1 and never redistributed. The warm-started large expert is so much better than nano/small
that the performance gradient overwhelms both the FLOPs penalty and load balance penalty.

This sweep isolates beta as the variable to find the regime where:
- Router entropy > 0 (routing diversity exists)
- Val TDLC BLER @ high SNR is still competitive with dense large

## Config

- Base: `conf/experiment/exp20_moe_phase2_beta_sweep.yaml`
- alpha=1e-3 (fixed), beta ∈ {0.5, 1.0, 2.0}
- 6k steps (2k frozen + 4k joint) — short enough to run 3 jobs in parallel
- Walltime: 3h per job

## Jobs

| beta | exp_name | Job ID | W&B run |
|------|----------|--------|---------|
| 0.5 | moe_phase2_b0p5_s67 | — | — |
| 1.0 | moe_phase2_b1p0_s67 | — | — |
| 2.0 | moe_phase2_b2p0_s67 | — | — |

## What To Watch

- `train/ema/router_entropy` — want >0.3 after unfreezing (step 2000+)
- `train/ema/expert_usage/{nano,small,large}` — want large <80%, meaningful nano/small share
- `val/tdlc/snr_bin_4/bler` @ SNR=17 — should stay ≤0.3 for the run to be worth continuing
- `val/tdlc/bler` overall — should not degrade more than ~0.05 above dense large (0.866)

## Results

| beta | Router entropy (step 6k) | large usage | TDLC BLER @ SNR=17 | Decision |
|------|--------------------------|-------------|---------------------|---------|
| 0.5 | — | — | — | — |
| 1.0 | — | — | — | — |
| 2.0 | — | — | — | — |

## Decision Criteria

- If entropy > 0.3 and BLER @ SNR=17 < 0.35 → promising, run to 12k
- If entropy > 0.3 but BLER too high → beta too strong, try intermediate
- If entropy ≈ 0 → beta still too weak, try even higher
