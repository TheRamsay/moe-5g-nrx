# MoE Phase 1 s56 v1

## Question

Does fixing the stem bottleneck (state_dim=32 → 56) restore waterfall performance in
joint-from-scratch MoE training?

The original Phase 1 run (exp12, `2026-04-06-moe-nl-phase1-v1`) used state_dim=32, which the
stem bottleneck study showed costs ~15pp BLER in the waterfall region. This reruns Phase 1 with
the correct state_dim=56, using the HF dataset for training throughput.

## Config

- Model: `moe_nl` (nano/small/large experts, state_dim=56)
- `alpha=1e-3`, `beta=0.1` (inherited from Phase 1 sweep winner)
- 10k steps, lr=1e-3, wd=1e-4, constant LR, seed=67
- HF dataset: `Vack0/moe-5g-nrx`, batch_size=128, hf_num_workers=4
- Validation: every 500 steps, 5 SNR bins per profile
- Experiment config: `conf/experiment/exp18_moe_phase1_s56_hf.yaml`

## Job

`18918948.pbs-m1.metacentrum.cz` — 12 CPUs, 1 GPU, 48 GB, 4h walltime  
W&B run: `2op33pak` (`moe_phase1_s56_a1e3_b0p1_s67_18918948`)

## Results

| Metric | Value | Notes |
|---|---|---|
| Best checkpoint step | 9000 | |
| Router entropy (EMA, final) | 0.294 | Healthy diversity early, collapsed late |
| nano usage (EMA, final) | 46% | |
| small usage (EMA, final) | 54% | |
| large usage (EMA, final) | ≈0% | Abandoned by router late in training |
| Realized FLOPs ratio (EMA) | 0.30 | 30% of dense large |
| Val TDLC BLER (overall) | 0.917 | vs dense large 0.866 |
| Val TDLC BLER @ SNR=17 | 0.588 | vs dense large ~0.275; s32 was ~1.0 |
| Val UMA BLER (overall) | 0.950 | |
| Val UMA BLER @ SNR=22 | 0.819 | |

Checkpoint artifact: `knn_moe-5g-nrx/moe-5g-nrx/model-moe_phase1_s56_a1e3_b0p1_s67-2op33pak:best`

## Interpretation

The s56 fix restores the waterfall — TDLC BLER@SNR=17 drops from ~1.0 (s32) to 0.588,
confirming the stem was the bottleneck in the original Phase 1.

However, the FLOPs penalty (alpha=1e-3) is too aggressive: by step 10k the router has
abandoned the large expert entirely (0% usage), settling on nano+small only at 30% FLOPs.
This saves compute but at the cost of BLER — 0.588 vs dense large's ~0.275 at high SNR.
The model found a cheap-but-suboptimal routing policy.

Neither Pareto-dominates dense large: lower FLOPs but significantly worse high-SNR BLER.

## Decision

Proceed to Phase 2 (warm-start + staged training) to test whether pre-trained experts and
router pre-training can prevent the router from collapsing away from large. If Phase 2 also
fails, next step is a beta/alpha sweep or capacity-constrained routing.
