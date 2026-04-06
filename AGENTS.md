# Context for Agents: Compute-Aware MoE for Efficient 5G Neural Receivers

## Goal

Build a **compute-aware 5G neural receiver** that keeps BLER close to a strong static dense baseline while reducing **average FLOPs** through adaptive routing.

This is a **domain-aware** project, not just "more layers": routing should use channel-quality information already present in the receiver input (received grid + LS estimate), and the main result should be **BLER vs Average FLOPs**.

## Current Project State

- **Data pipeline:** training uses on-the-fly **Sionna** generation; validation/test use cached `.pt` datasets.
- **Training distribution:** `mixed` = alternating `uma` and `tdlc` batches.
- **Validation / test policy:** cached `uma` and `tdlc` only; 7 SNR bins enabled by default in evaluation.
- **Important fix already applied:** channel power is normalized across profiles in `src/data/sionna_generator.py`, so nominal SNR is now comparable between `UMa` and `TDL-C`.

## Dense Baseline Status

- Static dense receiver is implemented and working in `src/models/dense.py`.
- Best current dense-capacity winner on validation/test is **large** (`exp05_dense_capacity_large`), with **mid** still close enough to matter for efficiency tradeoffs.
- One-batch overfit sanity check now exists and works (`training.overfit_single_batch=true`).

**Finalized dense checkpoints (20k steps, lr=1e-3, wd=1e-4, constant LR, seed=67):**

| Model | Params | val/tdlc BER | val/uma BER | best score | artifact |
|---|---:|---:|---:|---:|---|
| small | 168k | 0.1251 | 0.2751 | 0.2001 | `model-dense_small_final20k_constant_lr_s67-kivdz4qu:best` |
| mid | 306k | 0.1234 | 0.2738 | 0.1986 | `model-dense_mid_final20k_constant_lr_s67-jlnei9eg:best` |
| large | 450k | 0.1221 | 0.2683 | 0.1965 | `model-dense_large_final20k_constant_lr_s67-55l1dpby:best` |

All artifacts are under `knn_moe-5g-nrx/moe-5g-nrx/`. The large checkpoint is the canonical frozen
baseline. Small and mid were finalized to 20k steps specifically to give Phase 2 warm-start equal
footing across all three expert sizes (study: `2026-04-05-dense-small-mid-finalization-v1`).

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
- **MoE runs require `gpu-46gb.sh`** — training runs all 3 experts simultaneously (soft gating), 16 GB OOMs.
- Dense runs can use `gpu-16gb.sh`.
- For batch runs, checkpoints should go to `../artifacts/checkpoints` so they are synced back from scratch.

## MoE Direction

Architecture: **shared stem + channel-aware router + 3 heterogeneous experts**.

Expert family (width-heterogeneous, same depth):
- `small`: block_hidden_dim=32, readout_hidden_dim=96
- `mid`: block_hidden_dim=48, readout_hidden_dim=128
- `large`: block_hidden_dim=64, readout_hidden_dim=128

Training strategy:
1. **Phase 1 — Joint from scratch** (baseline, establishes alpha/beta hyperparameters)
2. **Phase 2 — Warm-started staged** (load dense checkpoints into experts, freeze → train router → unfreeze → joint fine-tune)
3. Compare Phase 1 vs Phase 2 on BLER vs FLOPs Pareto curve

Training routing:
- Gumbel-Softmax during training (`temperature=1.0`, `min_temperature=0.5`)
- Hard top-1 at inference

## Capacity & Architecture Findings (2026-04-06)

### Capacity Floor Study (`2026-04-06-dense-capacity-floor-v1`)
Backbone width/depth was varied while keeping the stem fixed at [64,64], state_dim=56.

| Model | Params | val TDLC BER | val UMA BER | best score (10k) | val TDLC BLER |
|---|---:|---|---|---|---|
| nano (block=8, 4 blk) | 90k | 0.1323 | 0.2814 | 0.2064 | 0.9711 |
| micro (block=16, 4 blk) | 104k | 0.1296 | 0.2791 | 0.2044 | 0.9492 |
| *small ref (20k)* | *168k* | *0.1251* | *0.2751* | *0.2001* | *0.9109* |
| *large ref (20k)* | *450k* | *0.1221* | *0.2683* | *0.1965* | *0.8660* |

**Key finding:** BER gap between nano and large is tiny (~1pp). BLER gap is more meaningful
(~8pp TDLC). The BER metric averages over low-SNR bins where all models fail equally,
masking the real signal. **Use BLER@high-SNR as the primary evaluation metric.**

### Stem Bottleneck Study (`2026-04-06-dense-stem-bottleneck-v1`)
Large backbone fixed (block_dim=64, 8 blocks). Only state_dim and stem_hidden_dims varied.

| Model | state_dim | val TDLC BER | val UMA BER | best score (10k) | TDLC ch. MSE |
|---|---:|---|---|---|---|
| stem_s32 | 32 | 0.1270 | 0.2762 | 0.2016 | 0.0785 |
| stem_s16 | 16 | 0.1454 | 0.2858 | 0.2156 | 0.2003 |
| *large ref (20k)* | *56* | *0.1221* | *0.2683* | *0.1965* | *0.0644* |

**Key finding:** state_dim=32 matches large with ~67% fewer stem FLOPs. state_dim=16 breaks —
channel_mse is 3× worse, indicating the bottleneck is **channel estimation capacity**, not
decoding. Safe working point: state_dim=32.

### Waterfall BLER Comparison (`2026-04-06-waterfall-compare-v1`)
Fine-grained (2 dB) SNR sweep across nano/small/large_s32/large_s56. Key finding: expert
size produces large BLER gaps **in the waterfall region only** — average BER/BLER masks this.

| SNR (dB) | nano | small | large_s32 | large_s56 |
|---|---|---|---|---|
| 13 | 0.996 | 0.998 | 0.989 | 0.969 |
| 15 | 0.922 | 0.834 | 0.779 | 0.649 |
| **17** | **0.722** | **0.548** | **0.436** | **0.284** |
| 19 | 0.452 | 0.310 | 0.244 | 0.105 |

Gap at SNR=17: nano→large_s56 = **44pp**. This justifies MoE routing.
UMA waterfall is much weaker (5-8pp gaps, waterfall not complete in eval range).

### Revised MoE Expert Design (Final)
The original expert range (block_dim 32/48/64) is too narrow. Redesigned with wider range
and **state_dim=56** (reverted from s32 — see rationale below):

| Expert | block_dim | num_blocks | readout_dim | state_dim |
|---|---:|---:|---:|---:|
| nano | 8 | 4 | 32 | 56 |
| small | 32 | 8 | 96 | 56 |
| large | 64 | 8 | 128 | 56 |

**Why state_dim=56, not 32:** In the MoE the stem is shared — state_dim doesn't affect
per-expert FLOPs differentiation. Routing nano vs large saves ~80% FLOPs with s56 vs ~82%
with s32 (negligible). But s32 costs 15pp BLER in the TDLC waterfall (large_s32 BLER=0.436
vs large_s56 BLER=0.284 at SNR=17). Bad trade. Use s56.

**FLOPs breakdown (s56, large model as reference):**
- Shared stem: 285M FLOPs (fixed, always paid)
- Nano expert: 35M FLOPs
- Large expert: 1319M FLOPs
- Total nano: 320M (20% of large total) → routing to nano saves **80% FLOPs**

### Nano Depth Study (`2026-04-06-dense-nano-depth-v1`)
Depth (2/4/8 blocks) barely matters at nano scale — differences are noise-level.
4 blocks is the confirmed choice for the nano expert.

### Warm-Start Checkpoints for Phase 2 (all s56, 20k steps, READY)

| Expert | Params | TDLC BLER@SNR17 | Artifact |
|---|---:|---:|---|
| nano | 90k | 0.674 | `model-dense_nano_final20k_constant_lr_s67-aos4hhid:best` |
| small | 168k | 0.548 | `model-dense_small_final20k_constant_lr_s67-kivdz4qu:best` |
| large | 450k | 0.284 | `model-dense_large_final20k_constant_lr_s67-55l1dpby:best` |

All three checkpoints use state_dim=56. Phase 2 can start immediately.

### SNR-Binned Validation Metrics
Training now logs per-SNR-bin BER/BLER/SER as timeseries during validation (every 500 steps).
Controlled by `validation.snr_bins` (default 5). Metric keys: `val/{profile}/snr_bin_{i}/bler`.
This makes the high-SNR waterfall shift visible during training without waiting for full eval.

## Phase 1 Experiment Status

**Confirmed findings:**
- `beta=0` → router collapses to large expert completely (entropy → 0)
- `beta=0.01` → too weak, still leans toward large (~50%)
- `beta=0.1` → prevents collapse (entropy ~0.88-0.91), all experts active
- `alpha ∈ {1e-5, 1e-4}` → negligible effect, need much larger alpha
- Per-profile routing differentiation visible even at 2k steps: TDLC prefers large/mid, UMA prefers small

**Locked hyperparameters for Phase 1:**
- `beta = 0.1` (load-balance penalty)
- `alpha = 1e-3` (sweep winner — see `experiments/2026-04-05-moe-alpha-sweep-v1/README.md`)

**Alpha sweep findings (jobs 18712100-18712102, finished):**
- `alpha=5e-3` → full router collapse (large ~0%), entropy 40% of max — threshold effect, not gradual
- `alpha=1e-2` → partial collapse (large ~29%), slow drift, entropy 55% of max
- `alpha=1e-3` → perfect routing (all ~33%), entropy 99.9%, best val loss
- BER nearly identical across all three; loss is the differentiator
- Phase 1 matches dense large baseline; no real compute savings yet (soft gating, all experts active)

**Phase 1 is complete. Original dense expert checkpoints are finalized. See Phase 2 redesign below.**

## Phase 2 (Current Direction — Ready to Start)

All warm-start checkpoints are finalized (see above). MoE NL Phase 1 confirmed routing
differentiates: TDLC→large (43%), UMA→nano (36%), overall entropy=0.85.

**Planned Phase 2 stages:**
1. Freeze experts → train router only (~2-3k steps)
2. Unfreeze → joint fine-tune (~10k steps) with alpha=1e-3, beta=0.1
3. Evaluate on **BLER@high-SNR vs realized FLOPs** Pareto curve

**Locked hyperparameters:** `alpha=1e-3`, `beta=0.1` (inherited from Phase 1).

## Original Phase 2 Setup (superseded)

**Goal:** warm-start MoE experts from finalized dense checkpoints, then train router + joint fine-tune.

**Expert checkpoints for warm-start:**
- `small`: `knn_moe-5g-nrx/moe-5g-nrx/model-dense_small_final20k_constant_lr_s67-kivdz4qu:best`
- `mid`: `knn_moe-5g-nrx/moe-5g-nrx/model-dense_mid_final20k_constant_lr_s67-jlnei9eg:best`
- `large`: `knn_moe-5g-nrx/moe-5g-nrx/model-dense_large_final20k_constant_lr_s67-55l1dpby:best`

**Rationale for equal-quality warm-start:** small and mid were re-trained to 20k steps (same recipe
as large) before Phase 2. Using the 10k capacity-sweep checkpoints would have created a quality
asymmetry biasing the router toward large from day one.

**Planned training stages:**
1. Freeze experts → train router only (~2-3k steps) so it learns to differentiate without disturbing pretrained weights
2. Unfreeze → joint fine-tune (~10k steps) with `alpha=1e-3`, `beta=0.1`

**Locked hyperparameters for Phase 2:** `alpha=1e-3`, `beta=0.1` (inherited from Phase 1).

## How This Compares To MEAN (van Bolderik et al., 2024)

MEAN is the closest prior work. Key differences:

| Aspect | MEAN | This project |
|---|---|---|
| Expert sizes | Homogeneous (same arch) | **Heterogeneous (different widths)** |
| Expert specialization | Assigned by SNR range at train time | **Emergent from gradient** |
| Router input | Raw received samples (simple MLP) | **Pooled shared stem features** |
| Training gate | Hard argmax + STE throughout | **Gumbel-Softmax → hard top-1** |
| Compute penalty | None | **α × expected_flops_ratio** |
| Channel models | CDL-C only | **Mixed UMA + TDL-C** |
| FLOPs accounting | Layer count (coarse) | **Explicit FLOPs model** |
| Pareto analysis | Not done | **BLER vs FLOPs curve** |

**Correction:** MEAN's gating network uses raw received samples as input, NOT oracle SNR. Do not claim oracle SNR routing as a MEAN limitation. The real differentiator is emergent vs assigned expert roles.

## Main Metrics To Care About

- `BER` and `BLER` on `uma` and `tdlc`
- `Average realized FLOPs` (at inference, hard top-1 — not expected FLOPs)
- `BLER vs realized FLOPs` Pareto curve
- Expert utilization by SNR bin and by profile
- Router entropy (collapse diagnostic)

## GPU Utilization & Training Data

On-the-fly Sionna generation is the training bottleneck — 6.9x slower than the GPU training
step on an A40 (483ms gen vs 70ms step). GPU sits idle ~87% of the time.

**CPU Sionna profiling results (job 18730965, finished):**

| Mode | Batch time | vs GPU step (70ms) |
|---|---:|---:|
| Sionna GPU (workers=0) | 483 ms | 6.9× |
| Sionna CPU (workers=0) | 699 ms | 10.0× |
| Sionna CPU (workers=2) | 546 ms | 7.8× |

**Conclusion: CPU Sionna does not help.** CPU gen is slower than GPU gen (699ms vs 483ms).
Even with workers=2, estimated GPU util ≈ 13% — same as today's on-the-fly setup (~15%).
The gen/step ratio (~8-10×) is simply too large for overlap to matter.

**Fix: cached training data.** 500k-sample dataset being generated at
`~/moe-5g-datasets/train-500k-v1/train/{uma,tdlc}.pt` (job 18730966, running).
This eliminates Sionna from the training loop entirely.
Storage: ~48 KB/sample → 500k samples ≈ 24 GB per profile.

## Key Rules For Future Work

- **Same dataset version** (`dense-v1`) across all MoE experiments for val/test
- **Same val/test protocol** (cached uma + tdlc, 7 SNR bins)
- **Same FLOP accounting** (realized FLOPs at inference for Pareto, not expected)
- **Compare MoE against tuned local dense baseline**, not paper numbers
- **Multi-seed (≥3)** only for final chosen model variants — not during exploration
- **Test set is locked** — run evaluate.py on test split exactly once per final model variant
- **state_dim=56** for all MoE experiments — do not use s32 (costs 15pp waterfall BLER)
