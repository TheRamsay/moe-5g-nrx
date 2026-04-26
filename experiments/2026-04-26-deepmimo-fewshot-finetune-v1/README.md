# DeepMIMO Few-Shot OOD Fine-Tune (v1)

**Status: drafted — not yet submitted. Fire after current β-warmup evals finish.**

## Question

Zero-shot OOD eval (`2026-04-26-deepmimo-ood-eval-v1`) showed **both dense and
MoE fail** on `asu_campus1` ray-traced channels (~0.99 BLER each). Does a
**brief fine-tune on a small slice of OOD data** recover performance?

If yes: synthetic-trained NRX is a **strong transfer base**. The compute-aware
MoE then needs only adaptation, not full re-training. **Major A+ upgrade.**

## Plan

### Stage 1: Generate small DeepMIMO TRAIN slice

Re-use `experiments/2026-04-12-deepmimo-ood-dataset-v1/submit.sh` with
`SPLIT=train` and a small `NUM_SAMPLES`. Saves Arrow dataset to
`data/train/asu_campus1/` and logs as W&B artifact
`dataset-train-asu_campus1:latest`.

```bash
SCENARIO=asu_campus1 SPLIT=train NUM_SAMPLES=2048 \
  bash experiments/2026-04-12-deepmimo-ood-dataset-v1/submit.sh qsub
```

Walltime: 30-60 min (CPU-heavy generation, 2k samples is fast).

### Stage 2: Fine-tune 2 checkpoints

Resume each from its existing best checkpoint, train briefly on
`asu_campus1` only with reduced LR.

Two new experiment YAMLs (`exp50_finetune_dense_large_ood.yaml`,
`exp51_finetune_exp26_ood.yaml`) override:

| Key | Value |
|---|---|
| `training.resume_from` | `<dense_large or exp26 :best artifact>` |
| `dataset.channel_profile` | `asu_campus1` (single profile, train_loader handles arbitrary names) |
| `training.hf_train_data_dir` | path to `data/train/` so loader reads `data/train/asu_campus1/` |
| `training.hf_max_samples` | `2048` |
| `training.max_steps` | `500` |
| `training.learning_rate` | `1e-4` (10× below base — standard fine-tune) |
| `model.compute.flops_penalty_alpha` | `2e-3` for exp26 (keep router incentive); `0` for dense |

Single submit_finetune.sh that fires 2 qsub jobs. ~30 min each.

### Stage 3: Eval on existing 32k OOD test

Same `evaluate.py` invocation as the OOD eval study, pointing at the
fine-tuned checkpoints. Two more qsub jobs, ~15 min each.

## Cluster total

- 1 generation job (~30-60 min CPU)
- 2 fine-tune jobs (~30 min each)
- 2 eval jobs (~15 min each)
- **Wall: ~2-3 h elapsed**

## Decisions made

- **N = 2048** (single slice — proves the concept). Sweep {256, 1k, 4k} only
  if 2k works.
- **500 steps** (brief — true few-shot, not full retrain).
- **lr = 1e-4** (standard fine-tune, 10× below base).
- **Fine-tune both dense_large AND exp26** (control + treatment).
- **exp31 (2-expert) excluded for now** — keeps scope tight; add only if N=2k
  result is positive.

## Decision criteria after Stage 3

| Outcome | Story for the report |
|---|---|
| Both recover to ~0.5 BLER | **Synthetic NRX is a strong transfer base.** Headline. |
| **MoE recovers FASTER (better BLER) than dense** | **Compute-aware routing transfers well — A+ punchline.** |
| Dense recovers, MoE doesn't | MoE overfits to ID routing — honest finding (still publishable). |
| Neither recovers (≥0.95 BLER) | Sionna→ray-tracing gap is fundamental — N=2k insufficient. Try N=10k or scope to "needs more data". |

## Jobs (to fill in after submission)

| Stage | Job ID | W&B run | Status |
|---|---|---|---|
| 1 — gen DeepMIMO TRAIN | _tbd_ | _tbd_ | not submitted |
| 2 — fine-tune dense_large | _tbd_ | _tbd_ | blocked on Stage 1 |
| 2 — fine-tune exp26 | _tbd_ | _tbd_ | blocked on Stage 1 |
| 3 — eval finetuned dense_large | _tbd_ | _tbd_ | blocked on Stage 2 |
| 3 — eval finetuned exp26 | _tbd_ | _tbd_ | blocked on Stage 2 |

## Results template

| Model | Setting | TDLC BLER | UMA BLER | OOD asu_campus1 BLER | OOD routing | OOD FLOPs % |
|---|---|---:|---:|---:|---|---:|
| Dense large | zero-shot | 0.866 | 0.936 | 0.990 | — | 100% |
| **Dense large** | **fine-tuned (N=2k, 500 steps)** | _tbd_ | _tbd_ | **_tbd_** | — | 100% |
| exp26 MoE | zero-shot | 0.867 | 0.937 | 0.992 | nano 75% / small 14% / large 11% | 32% |
| **exp26 MoE** | **fine-tuned (N=2k, 500 steps)** | _tbd_ | _tbd_ | **_tbd_** | _tbd_ | _tbd_ |

## Caveats / open risks

- **Catastrophic forgetting:** fine-tuning may degrade in-distribution (UMa+TDLC) BLER.
  Eval will catch this. If it happens, mention it as a tradeoff.
- **Validation:** we don't have a val split for `asu_campus1` to track during
  fine-tune. Use the train slice for early-stop monitoring (or skip val and
  pick last checkpoint).
- **Router instability under fine-tune:** the router sees a brand-new channel
  distribution; with α=2e-3 it might collapse to one expert quickly. If so,
  try α=0 (no FLOPs penalty during fine-tune) as a follow-up.
