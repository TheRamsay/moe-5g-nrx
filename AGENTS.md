# Context for Agents: Compute-Aware MoE for Efficient 5G Neural Receivers

## Goal

Build a **compute-aware 5G neural receiver** that keeps BLER close to a strong static dense baseline while reducing **average FLOPs** through adaptive routing.

This is a **domain-aware** project, not just "more layers": routing should use channel-quality information already present in the receiver input (received grid + LS estimate), and the main result should be **BLER vs Average FLOPs**.

## Current Project State (2026-04-13)

- **Best result:** Asymmetric warm-start 12k (exp23 + resume) — all 3 experts active
  (33% large, 29% nano, 38% small), val BLER 0.913 avg at 55% FLOPs. First genuinely
  heterogeneous routing with competitive quality.
- **Pareto frontier:** Phase 2 v1 (BLER 0.879 / 100% FLOPs) → Asym warm 12k (0.913 / 55%) → Phase 1 s56 (0.926 / 48%)
- **Data pipeline:** training loads from `Vack0/moe-5g-nrx` on HuggingFace
  (250k samples per profile, cached in persistent storage on the cluster).
  Validation/test use cached `.pt` datasets at `/auto/brno2/home/ramsay/moe-5g-datasets/dense-v1/`.
- **Training distribution:** `mixed` = alternating `uma` and `tdlc` batches.
- **Resume support:** `training.resume_from` accepts local path or W&B artifact ref.
  Restores model + optimizer + global_step. Scheduler steps forward automatically.
- **Checkpoint policy:** both "best" and "latest" checkpoints upload to W&B.
  Local checkpoints synced from `$WORK_ROOT/checkpoints/` during job cleanup.

## Dense Baseline Status

**Finalized dense checkpoints (20k steps, lr=1e-3, wd=1e-4, constant LR, seed=67):**

| Model | Params | val/tdlc BER | val/uma BER | best score | artifact |
|---|---:|---:|---:|---:|---|
| nano | 90k | 0.1323 | 0.2814 | 0.2064 | `model-dense_nano_final20k_constant_lr_s67-aos4hhid:best` |
| small | 168k | 0.1251 | 0.2751 | 0.2001 | `model-dense_small_final20k_constant_lr_s67-kivdz4qu:best` |
| large | 450k | 0.1221 | 0.2683 | 0.1965 | `model-dense_large_final20k_constant_lr_s67-55l1dpby:best` |

All artifacts are under `knn_moe-5g-nrx/moe-5g-nrx/`.

## MoE Architecture

**Shared stem + channel-aware router + 3 heterogeneous experts.**

| Expert | block_dim | num_blocks | readout_dim | state_dim | Params |
|---|---:|---:|---:|---:|---:|
| nano | 8 | 4 | 32 | 56 | 90k |
| small | 32 | 8 | 96 | 56 | 168k |
| large | 64 | 8 | 128 | 56 | 450k |

- Gumbel-Softmax during training (temperature=1.0, min_temperature=0.5)
- Hard top-1 at inference
- Loss: `BCE + channel_mse_weight * channel_MSE + alpha * expected_FLOPs + beta * load_balance`

**FLOPs breakdown (large model as reference = 1.604G):**
- Shared stem: 285M (always paid)
- Nano expert: 35M → total 320M (20% of large)
- Small expert: 557M → total 842M (52% of large)
- Large expert: 1319M → total 1604M (100%)

## MoE Experiment Results

### Phase 1 — Joint from scratch (exp18)
- All experts start random, trained jointly with router, 10k steps
- alpha=1e-3, beta=0.1, state_dim=56
- **Result:** router abandoned large (FLOPs penalty too aggressive), settled on nano+small
- Test eval: BLER 0.926 avg, 48% FLOPs
- Checkpoint: `model-moe_phase1_s56_a1e3_b0p1_s67-2op33pak:best`

### Phase 2 v1 — Warm-start + staged (exp17)
- Stem + all 3 experts warm-started from dense checkpoints
- 2k frozen (router only) + 10k joint, alpha=1e-3, beta=0.1
- **Result:** 100% large from step 1, never redistributed (router collapse)
- Test eval: BLER 0.879 avg, 100% FLOPs (functionally a fine-tuned dense large)
- Checkpoint: `model-moe_phase2_v1_a1e3_b0p1_s67-89no8f1k:best`

### Anti-Collapse Sweep

**What we tried to break Phase 2 router collapse:**

| Mechanism | Config | Result |
|---|---|---|
| β=0.5, β=1.0 | exp20 | Collapsed (MSE load balance too weak) |
| β=2.0 | exp20 | Forced-uniform 33/33/33 routing, terrible BLER (0.971 avg) |
| Soft capacity (cf=1.5, w=0.5) | exp21 | Diverse during frozen phase, collapsed after unfreeze |
| Switch aux (w=0.01) | exp22 | Total collapse, mechanism too weak at this weight |
| **Asymmetric warm-start** | **exp23** | **Positive result — see below** |

### Asym Warm 12k — Best Result (exp23 + resume)
- Stem + nano + small warm-started; **large starts random** (no warm-start)
- Single joint phase (no freezing), alpha=1e-3, beta=0.1
- 6k steps: large=0% (hadn't trained up), router used nano+small only
- **Extended to 12k via resume:** large caught up, router discovered it
- Final routing: 33% large, 29% nano, 38% small (all three active)
- Val: TDLC BLER=0.881, UMA BLER=0.944, avg=0.913, FLOPs=55%
- Val TDLC BLER@SNR=17: 0.411
- Checkpoint: `model-moe_phase2_asym_nlwarm_s67_12k-3witw8yw:best`

**Key insight:** asymmetric warm-start works because it removes the dominant-expert
trap. Large must earn its traffic by training up, rather than receiving it by default.
The router discovers large once it becomes competitive (~step 8000-10000).

### Opposite Failure Modes (Key Characterization Finding)

| Phase | What happens | Why |
|---|---|---|
| Phase 1 (cold start) | Router abandons large | FLOPs penalty dominates early when all experts are weak |
| Phase 2 (full warm-start) | Router locks on large | Warm-started large is strictly better than nano/small |
| Asym warm (partial warm-start) | Router discovers large gradually | Large must catch up, router has time to develop routing policy |

## Dataset Policy

- Reuse one cached dataset version per study phase, currently `dense-v1`.
- Canonical persistent layout:
  - `.../dense-v1/val/{uma,tdlc}.pt`
  - `.../dense-v1/test/{uma,tdlc}.pt`
- **IMPORTANT:** when submitting jobs, always include
  `validation.data_dir=/auto/brno2/home/ramsay/moe-5g-datasets/dense-v1/val`
  in RUN_ARGS — the default config has a relative path that doesn't work on scratch.

## Experiment Tracking

- Use **W&B** as the experiment registry.
- Use **artifacts** as the source of truth for checkpoints and datasets.
- Training/eval runs carry registry metadata (`study_slug`, `study_path`, `batch_name`, etc.).
- Evaluation logs structured tables (`eval/comparison`, `eval/snr_binned`, `eval/failures`).
- Study logs live under `experiments/` and should be kept up to date.

## Cluster Notes

- Default MetaCentrum path is through `scripts/metacentrum_job.sh` and the study `submit.sh` helpers.
- **Resources:** `ncpus=12:ngpus=1:mem=24gb:scratch_ssd=40gb` — no gpu_mem constraint.
- **Walltime:** 6k steps ≈ 5.8h, 12k steps ≈ 11.5h. Use 8h for 6k, 14h for 12k.
- **Do NOT specify `gpu_mem` in qsub** — model uses ~6GB VRAM, constraining GPU class only hurts queue time.
- Checkpoints are synced from `$WORK_ROOT/checkpoints/` during cleanup (survives walltime kills).
- Latest checkpoints also upload to W&B as artifacts.
- `deepmimov3` must be cached in the UV cache before first use (`uv sync` on login node).

## How This Compares To MEAN (van Bolderik et al., 2024)

MEAN and our work are **orthogonal contributions** that both apply MoE to 5G:

| Aspect | MEAN | This project |
|---|---|---|
| Goal | Per-SNR specialisation (quality) | **Compute efficiency (FLOPs savings)** |
| Expert sizes | Homogeneous (same arch) | **Heterogeneous (different widths)** |
| Expert specialization | Assigned by SNR range at train time | **Emergent from gradient** |
| Router input | Raw received samples (simple MLP) | **Pooled shared stem features** |
| Training gate | Hard argmax + STE throughout | **Gumbel-Softmax → hard top-1** |
| Compute penalty | None | **alpha * expected_flops_ratio** |
| Channel models | CDL-C only | **Mixed UMA + TDL-C** |
| Pareto analysis | Not done | **BLER vs FLOPs curve** |

**MEAN demonstrated MoE is applicable to 5G for specialisation purposes. Our work addresses
an orthogonal objective: compute efficiency via heterogeneous expert sizing.**

## Main Metrics To Care About

- `BER` and `BLER` on `uma` and `tdlc`
- `Average realized FLOPs` (at inference, hard top-1 — not expected FLOPs)
- `BLER vs realized FLOPs` Pareto curve
- Expert utilization by SNR bin and by profile
- Router entropy (collapse diagnostic)
- `BLER@SNR=17` on TDLC (waterfall region — most sensitive to expert quality)

## Key Rules For Future Work

- **Same dataset version** (`dense-v1`) across all MoE experiments for val/test
- **Same val/test protocol** (cached uma + tdlc, 7 SNR bins)
- **Same FLOP accounting** (realized FLOPs at inference for Pareto, not expected)
- **Compare MoE against tuned local dense baseline**, not paper numbers
- **Multi-seed (>=3)** only for final chosen model variants — not during exploration
- **Test set is locked** — run evaluate.py on test split exactly once per final model variant
- **state_dim=56** for all MoE experiments — do not use s32 (costs 15pp waterfall BLER)
- **HF training is the default** — set `training.hf_dataset=Vack0/moe-5g-nrx`
- **Always include `validation.data_dir`** in RUN_ARGS for cluster jobs
- **24GB RAM, no gpu_mem** for resource requests
- **8h walltime for 6k steps, 14h for 12k steps** — account for data loading stalls
