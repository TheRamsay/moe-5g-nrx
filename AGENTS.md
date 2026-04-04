# Context for Agents: Compute-Aware MoE for Efficient 5G Neural Receivers

## Goal

Build a **compute-aware 5G neural receiver** that keeps BLER close to a strong static dense baseline while reducing **average FLOPs** through adaptive routing.

This is a **domain-aware** project, not just "more layers": routing should use channel-quality information already present in the receiver input (received grid + LS estimate), and the main result should be **BLER vs Average FLOPs**.

## Current Project State

- **Data pipeline:** training uses on-the-fly **Sionna** generation; validation/test use cached `.pt` datasets.
- **Training distribution:** `mixed` = alternating `uma` and `tdlc` batches.
- **Validation / test policy:** cached `uma` and `tdlc` only, not cached `mixed` by default.
- **Important fix already applied:** channel power is normalized across profiles in `src/data/sionna_generator.py`, so nominal SNR is now comparable between `UMa` and `TDL-C`.

## Dense Baseline Status

- Static dense receiver is implemented and working in `src/models/dense.py`.
- Best current dense-capacity winner on validation/test is **large** (`exp05_dense_capacity_large`), with **mid** still close enough to matter for efficiency tradeoffs.
- One-batch overfit sanity check now exists and works (`training.overfit_single_batch=true`).

## Dataset Policy

- Reuse one cached dataset version per study phase, currently `dense-v1`.
- Canonical persistent layout:
  - `.../dense-v1/val/{uma,tdlc}.pt`
  - `.../dense-v1/test/{uma,tdlc}.pt`
- Do **not** regenerate val/test for every run unless simulator/data semantics changed.

## Experiment Tracking

- Use **W&B** as the experiment registry.
- Use **artifacts** as the source of truth for checkpoints and datasets.
- Training/eval runs carry registry metadata (`study_slug`, `study_path`, `batch_name`, etc.).
- Evaluation logs structured tables (`eval/comparison`, `eval/snr_binned`, `eval/failures`).
- Study logs live under `experiments/` and should be kept up to date.

## Cluster Notes

- Default MetaCentrum path is through `scripts/metacentrum_job.sh` and the study `submit.sh` helpers.
- Resource presets live in `experiments/resources/`.
- Recommended default is `gpu-16gb.sh` for dense runs.
- For batch runs, checkpoints should go to `../artifacts/checkpoints` so they are synced back from scratch.

## MoE Direction

- Use a **shared stem + channel-aware router + 3 heterogeneous experts**.
- Best practical expert family is the already-trained dense hierarchy:
  - small
  - mid
  - large
- Preferred training order:
  1. minimal **joint MoE** first,
  2. then warm-start / staged variant,
  3. then compute-penalty sweep for BLER-FLOPs tradeoff.
- Training routing:
  - Gumbel-Softmax during training
  - hard top-1 at inference

## Main Metrics To Care About

- `BER` and `BLER` on `uma` and `tdlc`
- `Average FLOPs`
- `BLER vs FLOPs`
- expert utilization by profile / SNR

## Key Rule For Future Work

Keep comparisons fair:

- same dataset version,
- same validation/test protocol,
- same FLOP accounting rules,
- compare MoE primarily against the **tuned local dense baseline**, not only against numbers quoted from papers.
