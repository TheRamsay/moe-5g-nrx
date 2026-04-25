# Context for Agents: Compute-Aware MoE for Efficient 5G Neural Receivers

## Goal

Build a **compute-aware 5G neural receiver** that keeps BLER close to a dense baseline while reducing **average FLOPs** via adaptive routing. Main result: **BLER vs Average FLOPs** Pareto curve. Router uses channel-quality features from the shared stem — not raw SNR.

## Current State (2026-04-25)

**Pareto frontier (test split):**

| Run | Avg BLER | FLOPs % | Notes |
|---|---|---|---|
| Dense large (baseline) | 0.901 | 100% | 450k params |
| Phase 2 v1 | 0.879 | 100% | router collapsed |
| **Asym warm 12k** | **0.910** | **61%** | **best result** |
| Phase 1 s56 | 0.926 | 48% | large abandoned |

**Best result:** Asym warm 12k — within 0.9 pp of dense large at 61% FLOPs. SNR-adaptive routing: TDLC 46% large, UMA 31% nano. Checkpoint: `model-moe_phase2_asym_nlwarm_s67_12k-3witw8yw:best`

**20k extension:** val TDLC BLER 0.851 at step 16k, routing stable ~39/22/39. Test eval pending.

**Alpha sweep in flight (2026-04-25):** 4-point Pareto curve `exp24..exp27`
(α ∈ {5e-4, 1e-3, 2e-3, 5e-3}), bs=128, 12k steps from scratch, asym warm-start,
seed=67. α=1e-3 (exp25) is an anchor re-baseline on the 50k subset (original
3witw8yw used full HF stream). Study folder:
`experiments/2026-04-25-moe-alpha-sweep-v1/`. Submit via `bash submit.sh qsub`.

## Dense Baselines (20k steps, lr=1e-3, wd=1e-4, seed=67)

| Model | Params | TDLC BLER | UMA BLER | Artifact |
|---|---:|---|---|---|
| nano | 90k | 0.971 | 0.961 | `model-dense_nano_final20k_constant_lr_s67-aos4hhid:best` |
| small | 168k | 0.911 | 0.951 | `model-dense_small_final20k_constant_lr_s67-kivdz4qu:best` |
| large | 450k | 0.866 | 0.936 | `model-dense_large_final20k_constant_lr_s67-55l1dpby:best` |

All under `knn_moe-5g-nrx/moe-5g-nrx/`.

## MoE Architecture

Shared stem (285M FLOPs, always paid) + channel-aware router + 3 heterogeneous experts:

| Expert | block_dim | blocks | Params | Total FLOPs | % of large |
|---|---:|---:|---:|---:|---:|
| nano | 8 | 4 | 90k | 320M | 20% |
| small | 32 | 8 | 168k | 695M | 43% |
| large | 64 | 8 | 450k | 1604M | 100% |

- Gumbel-Softmax training (temp 1.0→0.5), hard top-1 at inference
- Loss: `BCE + channel_mse_weight * MSE + alpha * expected_FLOPs + beta * load_balance`
- state_dim=56 for all MoE experiments (s32 costs 15pp waterfall BLER)

## Key Experiment Findings

**Phase 1 (cold start):** router abandons large — FLOPs penalty too aggressive early. BLER 0.926 / 48%.

**Phase 2 (full warm-start):** router locks on large — warm large is always better. BLER 0.879 / 100%.

**Asym warm-start (breakthrough):** stem + nano + small warm, large random init. Large must earn traffic by training up. Router discovers large at ~step 8-10k. All 3 experts active. Nano underutilised at realistic SNR (absorbs hopeless low-SNR traffic).

**Anti-collapse sweep:** β=2.0 forces 33/33/33 but kills BLER; capacity constraint collapses after unfreeze; Switch aux too weak. Asym warm-start is the only approach that worked.

## Data

**Training:** 50k-sample Array3D subset at `/storage/brno2/home/ramsay/moe-5g-datasets/train-50k-array3d/{uma,tdlc}`. Always set both `training.hf_dataset=Vack0/moe-5g-nrx` and `training.hf_train_data_dir=<path>` and `training.hf_max_samples=50000`.

**Val/test:** cached `.pt` at `/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/{val,test}/{uma,tdlc}.pt`. Always pass `validation.data_dir=/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/val` in RUN_ARGS.

**Training distribution:** `dataset=mixed` = alternating uma/tdlc batches.

## Cluster

- **Resources:** `ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb` — no `gpu_mem`
- **Walltime:** with optimized pipeline (~3k samples/s), 6k steps ≈ 1h, 20k steps ≈ 3-4h. Add buffer for startup + validation.
- **Default training config:** `batch_size=512`. Loader is single-process (`num_workers=0`); the worker config knobs were removed in commit `aed562e`.
- **Why num_workers=0:** the 22 GB Arrow training table leaves no headroom on a 32 GB node for forked/spawned worker copies (history of OOM, fork-lock, CoW blowup). For the MoE asym-warm step (~3k samples/s), `data_t` is negligible vs `step_t` so single-process is free. For the dense baseline step (~1150 samples/s), `data_t ≈ 0.21s` is ~46% of iter time — a real cost, but workers still aren't safe given RAM. Bigger win available: materialize Arrow → torch tensors at preload (eliminates per-batch HF gather), see CURRENT.md.
- Checkpoints sync from `$WORK_ROOT/checkpoints/` on cleanup; latest also uploads to W&B.
- Resume: `training.resume_from` accepts local path or W&B artifact ref.

## Metrics

- `BER` / `BLER` on `uma` and `tdlc`
- `Average realized FLOPs` at inference (hard top-1, not expected FLOPs)
- `BLER vs FLOPs` Pareto curve
- Expert utilization by SNR bin and profile; router entropy (collapse diagnostic)
- `BLER@SNR=17` on TDLC (waterfall region, most sensitive)

## Key Rules

- **Same dataset** (`dense-v1`) for all val/test across MoE experiments
- **Realized FLOPs** (inference, hard top-1) for Pareto — not expected FLOPs
- **Compare against local dense baseline** (not paper numbers)
- **Multi-seed (≥3)** only for final variants, not during exploration
- **Test set locked** — run evaluate.py once per final model variant
- **state_dim=56** for all MoE — do not use s32
- **Always include `validation.data_dir` and `training.hf_train_data_dir`** in cluster RUN_ARGS
- **No `gpu_mem` in qsub**

## A-Grade Roadmap (2026-04-25)

To turn the current B-level result into an A-level report. Bias toward what
makes the *report* defensible, not what's easy.

| # | Task | Status |
|---|---|---|
| 1 | Asym warm 20k test eval | pending |
| 2 | **Alpha sweep (4 jobs)** — exp24..exp27 in `2026-04-25-moe-alpha-sweep-v1` | submitted? track in study README |
| 3 | 3-seed run on winning α (s32, s42 alongside s67) | blocked on #2 |
| 4 | Random-feature router ablation (proves channel-aware claim) | not started |
| 5 | Doc cleanup: checkpoint_report §4 dataset description, archictures.png typo, routing-column granularity | not started |
| 6 | (A+ stretch) 2-expert ablation; DeepMIMO OOD eval | not started |

**Cut**: difficulty-guided routing, dataloader Arrow→torch refactor,
re-baselining dense at bs=512. None move the rubric.

## vs MEAN (van Bolderik et al., 2024)

MEAN: homogeneous experts, per-SNR specialisation, no compute penalty, CDL-C only. Our work: heterogeneous experts, compute efficiency via FLOPs penalty, emergent routing, mixed UMA+TDL-C, Pareto analysis. Orthogonal contributions.
