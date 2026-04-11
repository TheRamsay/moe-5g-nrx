# MoE Phase 2 v1 — Warm-start + Staged Training

## Question

Does warm-started staged training Pareto-dominate Phase 1 joint-from-scratch on BLER vs FLOPs?

Hypothesis: initialising each expert from its matching pre-trained dense checkpoint (nano/small/large)
and pre-training only the router for 2k steps (experts frozen) should prevent router collapse and
allow the router to learn channel-condition-aware routing before the experts start shifting.

## Config

- Model: `moe_nl` (nano/small/large experts, state_dim=56)
- `alpha=1e-3`, `beta=0.1`
- Staged: 2k steps experts frozen (router + stem only), then 10k steps joint — 12k total
- Warm-start: stem ← dense_large, each expert ← matching dense checkpoint
- HF dataset: `Vack0/moe-5g-nrx`, batch_size=128, hf_num_workers=4
- Validation: every 500 steps, 5 SNR bins per profile
- Experiment config: `conf/experiment/exp17_moe_phase2_v1.yaml`

## Warm-start Checkpoints

| Component | Source artifact |
|---|---|
| Stem | `model-dense_large_final20k_constant_lr_s67-55l1dpby:best` |
| Expert: nano | `model-dense_nano_final20k_constant_lr_s67-aos4hhid:best` |
| Expert: small | `model-dense_small_final20k_constant_lr_s67-kivdz4qu:best` |
| Expert: large | `model-dense_large_final20k_constant_lr_s67-55l1dpby:best` |

## Job

`18929182.pbs-m1.metacentrum.cz` — 12 CPUs, 1 GPU, 48 GB, 8h walltime  
W&B run: `89no8f1k` (`moe_phase2_v1_a1e3_b0p1_s67_18929182`)

## Results

| Metric | Value | Notes |
|---|---|---|
| Best checkpoint step | 12000 | Kept improving to end |
| Router entropy (EMA, final) | ≈0 | Full collapse throughout |
| nano usage (final) | ≈0% | |
| small usage (final) | ≈0% | |
| large usage (final) | 100% | Collapsed from frozen stage, never recovered |
| Realized FLOPs ratio | 1.0 | No compute savings |
| Val TDLC BLER (overall) | 0.837 | vs dense large 0.866 — improved |
| Val TDLC BLER @ SNR=17 | 0.215 | vs dense large ~0.275 — 22% better |
| Val UMA BLER (overall) | 0.930 | |
| Val UMA BLER @ SNR=22 | 0.763 | |

Checkpoint artifact: `knn_moe-5g-nrx/moe-5g-nrx/model-moe_phase2_v1_a1e3_b0p1_s67-89no8f1k:best`

## Val BLER @ SNR=17 (TDLC) over training

| Step | BLER |
|---|---|
| 500 | 0.275 |
| 3000 | 0.245 |
| 8000 | 0.229 |
| 12000 | 0.215 |

## Interpretation

The warm-started large expert is so much better than nano/small that the router
rationally routes 100% to large from the very first frozen-stage step and never
redistributes — even after unfreezing and with load balance penalty beta=0.1.

The model converged to a fine-tuned dense large rather than a MoE: BLER improves
steadily (0.215 @ SNR=17, 22% better than baseline) but at 100% FLOPs with zero
routing diversity. The Pareto hypothesis is **not confirmed**.

The warm-start recipe does work as a fine-tuning technique — the model beats the
original dense large baseline — but the routing mechanism failed entirely.

## Why Router Collapse Happened

- Warm-started large expert dominates from step 1 (already trained to near-optimal)
- Performance gradient >> load balance penalty (beta=0.1 insufficient)
- FLOPs penalty (alpha=1e-3) too weak to overcome the BLER cost of routing away from large
- Router is rational: always picking large minimises loss

## Next Steps

To fix router collapse:
1. **Stronger load balance**: increase beta (e.g. 0.5–1.0) to force traffic onto nano/small
2. **Capacity constraints**: hard cap on fraction routed to large (e.g. max 60%)
3. **Alpha/beta sweep**: find the regime where routing diversity and BLER are both acceptable
4. **Curriculum**: start with higher alpha/beta, anneal down as routing stabilises
