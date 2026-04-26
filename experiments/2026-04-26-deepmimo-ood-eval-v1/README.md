# DeepMIMO OOD Evaluation (v1)

## Question

Does the compute-aware MoE generalize to **out-of-distribution channels** —
ray-traced DeepMIMO scenarios — that were never seen during training?

This is the biggest A+ claim still untapped. All prior eval is on Sionna 3GPP
synthetic data (UMa + TDL-C); a reviewer can fairly ask "does this work on
realistic channels?" DeepMIMO ray-traced channels answer that question.

## Background

Training pipeline uses Sionna's standard 3GPP channel models (UMa stochastic,
TDL-C tapped-delay). DeepMIMO uses ray-tracing on real building geometries —
different statistical structure, different multipath, different correlations.

For the OOD claim:
- **Train: Sionna UMa + TDL-C, 50k samples, 16-QAM, SIMO 1×4, 3.5 GHz**
- **Eval: DeepMIMO ray-traced channels, same modulation/antenna/carrier**

## Plan

Two stages:

### Stage 1: Generate DeepMIMO OOD test set
- Scenario: `asu_campus1` (outdoor university campus, 3.5 GHz — closest to UMa)
- 32,768 samples (matches Sionna test split size)
- Saved to `data/test/deepmimov3/asu_campus1/` as Arrow dataset
- Logged to W&B as `dataset-test-deepmimov3_asu_campus1:latest`
- Reuses `experiments/2026-04-12-deepmimo-ood-dataset-v1/submit.sh` (already in repo)

### Stage 2: Run evaluate.py with profiles=[uma, tdlc, deepmimo] on key checkpoints

| Checkpoint | Why | Source |
|---|---|---|
| dense_large | Mandatory baseline (full-compute reference) | `model-dense_large_final20k_constant_lr_s67-55l1dpby:best` |
| **exp26 (3-expert α=2e-3)** | **Headline MoE** — does it generalize? | `model-moe_alphasweep_asym_a2e3_s67-t6lkdep2:best` |
| exp24 (collapsed = dense large) | Sanity check (should match dense_large baseline) | `model-moe_alphasweep_asym_a5e4_s67-_:best` (artifact may be missing — wandb-flake) |
| exp31 (2-expert) | Does the 2-expert variant generalize differently? | `model-moe_ablation_2expert_a2e3_s67-5c0kshem:best` |

Each eval is 1 job, ~15 min (inference only, 32k samples × 3 profiles).

## Cluster

- Generation: 1 job, ~40 min walltime (CPU-bound), ncpus=4 mem=16gb
- Eval: 4 jobs, ~30 min walltime each, ncpus=4 ngpus=1 mem=16gb

## Submission

```bash
# Stage 1: generate the OOD dataset (one-time)
SCENARIO=asu_campus1 bash experiments/2026-04-12-deepmimo-ood-dataset-v1/submit.sh qsub

# Wait for Stage 1 to finish, then Stage 2:
bash experiments/2026-04-26-deepmimo-ood-eval-v1/submit_eval.sh qsub
```

## Decision Criteria

| Outcome | Interpretation |
|---|---|
| MoE BLER on DeepMIMO close to dense large | **Positive: model generalizes.** Headline A+ claim. |
| MoE BLER craters; dense_large also craters | Both models fail OOD — synthetic-trained NRX has limits. Honest scope statement. |
| MoE craters; dense_large holds | MoE overfits to Sionna; nano/small experts can't handle ray-tracing. Reframe story. |
| MoE FLOPs ratio shifts (e.g., router uses large more on OOD) | **Bonus claim:** router adapts compute to channel difficulty even on OOD. Reinforces channel-aware story. |

## Jobs

| Stage | Job ID | Status |
|---|---|---|
| Generate asu_campus1 | _tbd_ | pending |
| Eval dense_large | _tbd_ | blocked on generation |
| Eval exp26 | _tbd_ | blocked on generation |
| Eval exp31 | _tbd_ | blocked on generation |

## Results — to fill in after eval

| Checkpoint | UMa BLER | TDL-C BLER | **DeepMIMO BLER** | DeepMIMO routing l/n/s | DeepMIMO FLOPs % |
|---|---:|---:|---:|---|---:|
| dense_large | 0.936 | 0.866 | _tbd_ | — | 100% |
| **exp26** | 0.937 | 0.867 | _tbd_ | _tbd_ | _tbd_ |
| exp31 (2-expert) | 0.946 | 0.876 | _tbd_ | _tbd_ | _tbd_ |
