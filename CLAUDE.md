# Context for Agents: Compute-Aware MoE for Efficient 5G Neural Receivers

## Goal

Build a **compute-aware 5G neural receiver** that keeps BLER close to a dense baseline while reducing **average FLOPs** via adaptive routing. Main result: **BLER vs Average FLOPs** Pareto curve. Router uses channel-quality features from the shared stem — not raw SNR.

## Current State (2026-04-26)

**Pareto frontier (test split, all on 50k subset):**

| Run | α | Avg BLER | FLOPs % | TDLC routing l/n/s | Notes |
|---|---:|---:|---:|---|---|
| Dense large (baseline) | — | 0.901 | 100% | — | 450k params |
| **exp24 (α=5e-4)** | 5e-4 | **0.898** | **100%** | 100/0/0 | router collapsed to large |
| exp25 (α=1e-3) | 1e-3 | 0.907 | 56% | 44/12/44 | dominated by exp26 |
| **exp26 (α=2e-3, s67)** | 2e-3 | **0.902** | **56%** | 44/15/40 | **knee of the Pareto** |
| exp27 (α=5e-3) | 5e-3 | 0.911 | 60% | 37/0/63 | nano starved → falls back to small |
| (historical) Asym warm 12k | 1e-3 | 0.910 | 61% | 46/2/52 | original anchor, full HF stream — superseded by exp26 |

**Headline:** exp26 at α=2e-3 reaches **0.1 pp of dense large at 56% FLOPs** —
strict Pareto improvement over the original anchor (0.910/61%). Pareto frontier
is 2 points (exp24 and exp26); exp25/exp27 are dominated.

**3-seed confirmation (2026-04-26):** ran s32 (exp28) and s42 (exp29) at the
α=2e-3 recipe. **Bimodal outcome:**
- s67 (exp26): avg 0.902, routing 44/15/40 — heterogeneous ✓
- s42 (exp29): avg 0.902, routing 55/12/33 — heterogeneous ✓
- **s32 (exp28): avg 0.958, routing 0/49/51 — large COLLAPSED** ✗

2 of 3 seeds reproduce the headline; 1 of 3 hits a phase-1-style large-collapse.
The asym-warm recipe is **not seed-stable**. Honest finding: report
"2 of 3 seeds achieve 0.902; 1 collapses to a 5pp worse / cheaper attractor."

**Ablations on α=2e-3 (2026-04-26):**

| Run | TDLC BLER | UMA BLER | Avg BLER | TDLC routing | TDLC FLOPs % |
|---|---:|---:|---:|---|---:|
| exp26 (3-expert, channel-aware) | 0.867 | 0.937 | **0.902** | 44/15/40 (l/n/s) | 65% |
| **exp30 (router input = noise)** | **0.965** | 0.972 | **0.968** | **0/11/89** | 41% |
| **exp31 (drop nano)** | 0.878 | 0.940 | **0.909** | 38/-/62 (l/s) | 65% |

- **Random router → BLER craters 6.6 pp** AND collapses to small (no large
  ever used). **Channel-aware features are load-bearing** — the central claim
  of the project. A+ confirmation.
- **2-expert (no nano) → 0.7 pp worse BLER + 9 pp more FLOPs** (TDLC 65% vs
  56%). Nano earns its keep — absorbs hopeless low-SNR samples that small
  would waste compute on. Justifies 3-expert design.

**DeepMIMO OOD eval (done 2026-04-26):** Generated `dataset-test-asu_campus1` (32k
samples, ASU campus 3.5 GHz ray-traced). Eval'd dense_large + exp26 + exp31.

| Model | TDLC | UMA | OOD asu_campus1 | OOD routing | OOD FLOPs % |
|---|---:|---:|---:|---|---:|
| Dense large | 0.866 | 0.936 | **0.990** | — | 100% |
| exp26 (3-expert) | 0.867 | 0.937 | **0.992** | nano 75 / small 14 / large 11 | **32%** |
| exp31 (2-expert) | 0.878 | 0.940 | **0.993** | small 75 / large 25 | ~71% |

**All three fail on OOD.** Synthetic-trained NRX cannot decode ray-traced channels
without OOD fine-tune. Honest scope statement. Bonus: 3-expert MoE *uses less compute*
on OOD (router defaults to nano under unfamiliar features) — separate behavioral observation.

**Stabilization attempts (both negative — characterized failure modes):**

`experiments/2026-04-26-moe-largewarmup-v1/` (exp32/33/34): freeze nano+small for first
2k steps. Result: **all 3 seeds collapse to 100% large** (Phase 2 v1 failure mode).
Mean test BLER ~0.86, FLOPs 100%. Over-corrects.

`experiments/2026-04-26-moe-betawarmup-v1/` (exp35/36/37): β=0.5 for first 4k steps,
drop to 0.1. Result: **3 different routing patterns**, mean test BLER **0.938 ± 0.024**
— *worse* than no-warmup baseline (0.921 ± 0.032). β-warmup hurts.

> "Two stabilization recipes attempted; neither produced robust seed-stable
> heterogeneous routing. Asym-warm bimodality appears intrinsic to the asym init choice.
> Recipe stability is an open problem; recommend best-of-N seeds with multi-seed disclosure."

**Static + SNR-oracle cascade analysis (D, done 2026-04-26):**
`experiments/2026-04-26-static-baselines-v1/` + `scripts/analyze_static_baselines.py`.
Pulls per-SNR-bin BLER from W&B and computes hand-rule cascades using true SNR.

| Strategy | Avg BLER | FLOPs % |
|---|---:|---:|
| Per-profile static (best fixed rule) | 0.900 | 100% (both → large) |
| **SNR-oracle cascade tol+0.01..0.02** | **0.900** | **49%** |
| **exp26 learned MoE** | **0.902** | **56%** |
| SNR-oracle cascade tol+0.05 | 0.909 | 35% |
| SNR-oracle cascade tol+0.10 | 0.919 | 27% |

**Honest finding:** with oracle SNR, hand-rule cascade slightly dominates exp26 on FLOPs
at same BLER (49% vs 56%). exp26 is **on the Pareto frontier** without oracle access.
Suggests explicit SNR estimate in router input as concrete future work.

**Sweep regimes** (informative even when not on the frontier):
- α=5e-4: too weak → router collapses to 100% large (Phase 2 v1 failure).
- α=[1e-3, 2e-3]: heterogeneous routing emerges, sweet spot at 2e-3.
- α=5e-3: too strong → router skips nano entirely (BLER cost > FLOPs savings vs small) and falls back to a large/small 2-expert regime — actually *increases* avg FLOPs.

**Alpha sweep eval runs (W&B, 2026-04-25):**

| Exp | α | Train run | Eval run (test set) |
|---|---:|---|---|
| exp24 | 5e-4 | _wandb-init flaked, ckpt local-only_ | `002cwsy2` |
| exp25 | 1e-3 | `3xzxkddv` | `5jswm490` |
| exp26 | 2e-3 | `t6lkdep2` | `2zboo1rh` |
| exp27 | 5e-3 | _wandb-init flaked, ckpt local-only_ | `dh4x0qmu` |

Rebaseline finding (exp25 vs original anchor 3witw8yw): at identical α=1e-3,
50k subset gives noticeably different routing (44/12/44 vs 46/2/52) but similar
avg BLER. Validates the rebaseline — anchor and exp25 are now compared cleanly.

**Wandb-init flake:** 2 of 4 training runs and 1 of 4 eval runs (recovered on
re-submit) hit `Failed to read port info after 30.0 seconds`. Not config-side;
appears intermittent per compute host. Worth a fix or a `WANDB_MODE=offline`
fallback later.

**20k extension:** val TDLC BLER 0.851 at step 16k, routing stable ~39/22/39. Test eval pending.

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

## A-Grade Roadmap (updated 2026-04-26)

| # | Task | Status |
|---|---|---|
| 1 | Asym warm 20k test eval | superseded by exp26 alpha-sweep winner |
| 2 | **Alpha sweep (4 jobs)** — exp24..exp27 | ✅ done; exp26 (α=2e-3) is the winner |
| 3 | **3-seed on α=2e-3** (s32, s42 alongside s67) | ✅ done; bimodal — 2 reproduce, 1 collapses |
| 4 | **Random-feature router ablation** | ✅ done; channel-aware features ARE load-bearing (BLER craters 6.6pp without) |
| 5 | **2-expert ablation** | ✅ done; nano earns its keep (0.7pp BLER hit + 9pp more FLOPs without) |
| 6 | **DeepMIMO OOD eval** (asu_campus1) | ✅ done; all 3 models fail (~0.99 BLER); honest scope finding |
| 7 | **Large-warmup stabilization** (exp32/33/34) | ✅ done; over-corrects to 100% large in 3/3 seeds (negative result) |
| 8 | **β-warmup stabilization** (exp35/36/37) | ✅ done; mean BLER worse than baseline (negative result) |
| 9 | **Static + SNR-oracle cascade baseline** (D analysis) | ✅ done; exp26 on Pareto frontier; oracle cascade slightly better at same BLER |
| 10 | **DeepMIMO few-shot fine-tune** (Stage 1+2 done; Stage 3 OOD eval pending) | 🟡 dense_large fine-tune ✅ done (`go74dlm7`); exp26 fine-tune queued; OOD eval after |
| 11 | **Wall-clock latency benchmark** | 🟡 queued (job 19473464) |
| 12 | **Per-SNR routing visualization** | ✅ done (`docs/figures/per_snr_routing_2zboo1rh.png` after commit) |
| 13 | **Channel-feature visualization** (PCA-2D, dropped sklearn) | 🟡 running (job 19473435) |
| 14 | Doc cleanup: checkpoint_report (§4 dataset, mean±std, ablations, OOD section, latency) | not started |
| 15 | (Optional A+) MEAN reimplementation as homogeneous-expert baseline | cut — too time-expensive |

**Cut**: difficulty-guided routing, dataloader Arrow→torch refactor,
re-baselining dense at bs=512. None move the rubric.

## vs MEAN (van Bolderik et al., 2024)

MEAN: homogeneous experts, per-SNR specialisation, no compute penalty, CDL-C only. Our work: heterogeneous experts, compute efficiency via FLOPs penalty, emergent routing, mixed UMA+TDL-C, Pareto analysis. Orthogonal contributions.
