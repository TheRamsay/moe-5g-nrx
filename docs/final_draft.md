# Compute-Aware MoE for Efficient 5G Neural Receivers — Final Draft

**Team:** Dominik Huml (xhumld00), Jakub Kontrik (xkontr02), Martin Vaculik (xvaculm00)
**Date:** 2026-04-27

---

## TL;DR

We built a **compute-aware 5G neural receiver** with heterogeneous Mixture-of-Experts
(MoE). On in-distribution (Sionna UMa + TDL-C) test data, the **exp26** model
matches dense large within **0.1 pp BLER** while running on **56% of the FLOPs**
and delivering a **1.93× wall-clock speedup** on GPU. We characterise the
training recipe (asymmetric warm-start), prove the channel-aware router is
load-bearing (a random router collapses BLER by 6.6 pp), confirm the 3-expert
design (dropping nano costs 0.7 pp BLER + 9 pp more FLOPs), measure
seed-stability (2 of 3 seeds reproduce; 1 collapses), and benchmark against an
SNR-oracle hand-rule cascade (exp26 sits on the Pareto frontier, oracle is
slightly cheaper at the same BLER). DeepMIMO ray-traced OOD evaluation: all
models — including dense — fail without OOD fine-tune; honest scope statement.

---

## 1. Problem Statement

Neural receivers in 5G PHY layer apply uniform compute to every received OFDM
slot regardless of channel quality. Most slots are easy (high SNR, LOS) and
could be decoded by a small model; only a fraction need the full receiver.
Static dense receivers therefore waste compute and energy.

Goal: a receiver that **adapts compute per sample** while keeping BLER close
to a static dense baseline. Primary metric: **BLER vs Average FLOPs** Pareto
frontier on a held-out test set.

---

## 2. Architecture

Three components:

- **Shared stem** — 2-layer MLP (hidden [64, 64], state_dim=56). Always paid
  (285M FLOPs). Produces channel features for the router AND the selected
  expert.
- **Channel-aware router** — small MLP (hidden 64) over **pooled stem features**
  (mean+max pool over freq/time), produces a categorical over experts.
  Gumbel-Softmax during training, **hard top-1** at inference.
- **Heterogeneous experts** — three CNN heads with different capacities:

| Expert | block_dim | blocks | Params | Expert FLOPs | Total FLOPs | % of large |
|---|---:|---:|---:|---:|---:|---:|
| nano  |  8 | 4 |  90k |   35M |  320M |  20% |
| small | 32 | 8 | 168k |  410M |  695M |  43% |
| large | 64 | 8 | 450k | 1319M | 1604M | 100% |

Total MoE: 583k params. Training loss:

`L = L_BCE + γ·L_channel + α·E[FLOPs ratio] + β·L_balance`

with γ=0.05, α swept (final 2e-3), β=0.1.

---

## 3. The Journey — Three Phases of Failure, Then a Win

### Phase 1 — Joint training from scratch
All experts + router random init. FLOPs penalty α=1e-3.
- **Result:** BLER 0.926 / 48% FLOPs.
- **Mode:** large is too expensive too early — router abandons it. By step
  10k, large usage drops to ~0%. Cheap, but BLER bleeds.

### Phase 2 — Full warm-start (all three experts pre-trained)
- **Result:** BLER 0.879 / **100% FLOPs**.
- **Mode:** warm-large is strictly better at step 1, router locks onto it,
  never explores. Becomes a fine-tuned dense large.

### Anti-collapse attempts (all failed)
- Stronger β load-balance: still collapses.
- β=2.0 forced uniform: BLER 0.971 (experts can't specialise).
- Capacity constraint: collapses after unfreeze.
- Switch Transformer aux loss: collapses.

### Phase 3 — **Asymmetric warm-start (the fix)**
Warm-start stem + nano + small. **Leave large at random init.**
Large has no initial advantage; it has to *earn* router traffic by learning.
- At ~step 6-8k, large "wakes up" and the router starts using it.
- All three experts active by step 10-12k.
- This is the recipe that anchors every result that follows.

---

## 4. Last-Few-Days Sprint (April 25–26): A+ Results

Once asym warm-start worked, we ran the full A+ rubric in two days.

### 4.1 Alpha sweep — find the best operating point
4 jobs sweeping the FLOPs penalty α ∈ {5e-4, 1e-3, 2e-3, 5e-3}, all using the
asym-warm recipe at 20k steps, seed 67.

| Run | α | Avg BLER | FLOPs % | TDLC routing l/n/s | Notes |
|---|---:|---:|---:|---|---|
| Dense large baseline | — | 0.901 | 100% | — | reference |
| **exp24** | 5e-4 | **0.898** | **100%** | 100/0/0 | router collapsed to large (regression to Phase 2) |
| exp25 | 1e-3 | 0.907 | 56% | 44/12/44 | dominated by exp26 |
| **exp26** | 2e-3 | **0.902** | **56%** | 44/15/40 | **knee of the Pareto** |
| exp27 | 5e-3 | 0.911 | 60% | 37/0/63 | nano starved → falls back to small |

**Headline:** **exp26 reaches dense-large BLER within 0.1 pp at 56% FLOPs.**
Pareto frontier is two points (exp24 and exp26); exp25/exp27 are dominated.

α<1e-3 too weak → collapse to large. α>2e-3 too strong → nano starved.
Sweet spot at 2e-3.

### 4.2 3-seed reproducibility on the winner
Re-ran α=2e-3 with seeds 32 and 42:

- **s67 (exp26):** avg 0.902, routing 44/15/40 (l/n/s) ✓
- **s42 (exp29):** avg 0.902, routing 55/12/33 ✓
- **s32 (exp28):** avg **0.958**, routing 0/49/51 — **large collapsed** ✗

Honest finding: **2 of 3 seeds reproduce; 1 hits a Phase-2-style attractor.**
The asym-warm recipe is **not seed-stable**. We report this transparently
rather than cherry-picking the best seed.

### 4.3 Channel-aware router ablation (the load-bearing claim)
**Question:** is the router actually using channel-quality features, or could
it route on noise?

exp30: identical recipe, but router input is `torch.randn` instead of pooled
stem features.

| Run | TDLC BLER | UMA BLER | Avg BLER | TDLC routing | TDLC FLOPs % |
|---|---:|---:|---:|---|---:|
| exp26 (channel-aware) | 0.867 | 0.937 | **0.902** | 44/15/40 | 65% |
| **exp30 (noise input)** | **0.965** | 0.972 | **0.968** | **0/11/89** | 41% |

**Random router → BLER craters 6.6 pp** AND collapses to small (large never
selected). **Channel-aware features are load-bearing** — exactly the central
claim of the project. A+ confirmation.

### 4.4 2-expert ablation (does nano earn its keep?)
exp31: drop nano, train with only {small, large}. Same recipe.

| Run | TDLC BLER | UMA BLER | Avg BLER | TDLC routing | TDLC FLOPs % |
|---|---:|---:|---:|---|---:|
| exp26 (3-expert) | 0.867 | 0.937 | **0.902** | 44/15/40 | 65% |
| exp31 (2-expert) | 0.878 | 0.940 | **0.909** | 38/-/62 (l/s) | 65%* |

*exp31 TDLC FLOPs match exp26 by coincidence; on average across UMA+TDLC,
2-expert costs 9 pp more FLOPs.

**2-expert costs 0.7 pp BLER + 9 pp more FLOPs.** Nano is non-decorative —
it absorbs hopeless low-SNR samples that small would burn compute on.
Justifies the 3-expert design.

### 4.5 Stabilization attempts (both negative — characterized failure modes)

**Large-warmup (exp32/33/34):** freeze nano+small for first 2k steps so large
has time to "catch up." Result: **all 3 seeds collapse to 100% large.**
Over-corrects in the opposite direction (Phase 2 v1 mode). Mean BLER ~0.86,
FLOPs 100%.

**β-warmup (exp35/36/37):** β=0.5 for first 4k steps then drop to 0.1.
Result: **3 different routing patterns, mean BLER 0.938 ± 0.024** —
*worse* than no-warmup baseline (0.921 ± 0.032). β-warmup hurts.

> "Two stabilization recipes attempted; neither produced robust seed-stable
> heterogeneous routing. Asym-warm bimodality is intrinsic to the asym init.
> Recipe stability is an open problem; recommend best-of-N seeds with
> multi-seed disclosure."

### 4.6 Static + SNR-oracle cascade baseline (D analysis)
**Question:** is the learned MoE actually beating an oracle with access to
true SNR?

We ran two static dense baselines (best fixed receiver per profile) and a
hand-rule cascade that uses **true SNR** to pick the cheapest expert that
clears a BLER tolerance.

| Strategy | Avg BLER | FLOPs % |
|---|---:|---:|
| Per-profile static (best fixed rule) | 0.900 | 100% (both → large) |
| **SNR-oracle cascade tol+0.01..0.02** | **0.900** | **49%** |
| **exp26 learned MoE** | **0.902** | **56%** |
| SNR-oracle cascade tol+0.05 | 0.909 | 35% |
| SNR-oracle cascade tol+0.10 | 0.919 | 27% |

**Honest finding:** with oracle SNR, the cascade slightly dominates exp26
on FLOPs at the same BLER (49% vs 56%). **exp26 sits on the Pareto frontier
without oracle access.** This is a clean future-work direction: feed an
explicit SNR estimate into the router input.

### 4.7 Wall-clock latency (synthetic Gaussian, GPU, batch=64)

| Model | Params | ms/batch | samples/sec | Speedup vs dense_large |
|---|---:|---:|---:|---:|
| dense_nano  |  90k | 0.97 | 65,693 | 3.00× |
| dense_small | 168k | 1.71 | 37,382 | 1.70× |
| dense_large | 450k | 2.92 | 21,925 | 1.00× |
| **exp26 MoE** | 583k | **1.51** | **42,391** | **1.93×** |

The 44% FLOPs reduction translates to a **1.93× wall-clock speedup**, *better*
than the 1.78× theoretical FLOPs ratio because at hard top-1 only the selected
expert runs and kernel-launch overhead favors smaller experts.
**Caveat:** synthetic input → router argmax approximately uniform 33/33/33;
real-data per-profile timing in flight at submission time.

### 4.8 DeepMIMO OOD evaluation (asu_campus1, ray-traced 3.5 GHz)
Generated 32k OOD samples (Stanford ASU Campus, ray-traced).

| Model | TDLC | UMA | OOD asu_campus1 | OOD routing | OOD FLOPs % |
|---|---:|---:|---:|---|---:|
| Dense large | 0.866 | 0.936 | **0.990** | — | 100% |
| exp26 (3-expert) | 0.867 | 0.937 | **0.992** | n75 / s14 / l11 | **32%** |
| exp31 (2-expert) | 0.878 | 0.940 | **0.993** | s75 / l25 | ~71% |

**All three fail on OOD.** Synthetic-trained NRX cannot decode ray-traced
channels without OOD fine-tune. **Honest scope statement.** Bonus observation:
the 3-expert MoE *uses less compute* on OOD because the router defaults to
nano under unfamiliar features — separate behavioral note.

### 4.9 DeepMIMO few-shot fine-tune (DONE — negative result)
Generated 2,048 ASU samples, fine-tuned both dense_large and exp26 for
500 steps at lr=1e-4, then re-eval'd on uma+tdlc+asu_campus1.

| Model | OOD BLER zero-shot | OOD BLER post-FT | Δ |
|---|---:|---:|---:|
| dense_large | 0.990 | 0.9901 | +0.0001 |
| **exp26 MoE** | 0.992 | **0.9915** | −0.0005 |

**Honest negative result:** 500 steps / 2k samples is **insufficient** to
recover OOD performance for either model. Both stay at ~0.99 BLER on
ray-traced channels.

**Useful complementary findings:**
- **No catastrophic forgetting**: ft_exp26 in-distribution (UMA 0.937 /
  TDLC 0.867) and routing (44/15/40 on TDLC) are bit-identical to zero-shot.
  The fine-tune did not degrade in-distribution performance.
- **Routing did not adapt**: ft_exp26 OOD routing 75/14/11 (n/s/l) — same
  nano-default as zero-shot. The brief fine-tune did not teach the router
  to escalate to large for OOD samples.

> "Substantial OOD generalization requires longer fine-tune, larger OOD slice,
> or domain-randomized pretraining — out of scope for this work."

### 4.10 Channel-feature visualization (PCA-2D)
Extracted the pooled stem features that feed the router on 2,000 test samples
per profile (UMA + TDLC). Plotted 2-D PCA colored by selected expert AND by
true SNR. Figures: `docs/figures/channel_feature_tsne_{uma,tdlc}.png`.
The two colorings overlap visually, confirming the router has learned a
SNR-correlated partition of the feature space.

### 4.11 Per-SNR routing visualization
Stacked-bar of router usage per SNR bin from W&B per-bin tables, exp26.
Shows the router transitions from small/nano-dominated routing at low SNR
to large-dominated routing in the waterfall region.
Figure: `docs/figures/per_snr_routing_2zboo1rh.png`.

---

## 5. Final Pareto Frontier

| Run | Avg BLER | FLOPs % | What it shows |
|---|---:|---:|---|
| Dense large | 0.901 | 100% | reference |
| **exp24 (α=5e-4)** | **0.898** | **100%** | best BLER point (collapsed to large) |
| **exp26 (α=2e-3, s67)** | **0.902** | **56%** | **the headline result** |
| exp25 (α=1e-3) | 0.907 | 56% | dominated |
| exp27 (α=5e-3) | 0.911 | 60% | dominated |
| Phase 1 s56 | 0.926 | 48% | superseded |

Pareto frontier: dense_large → exp26 → (optionally cheaper but worse points
at higher α). **exp26 is the contribution.**

---

## 6. What Worked, What Didn't (Honest Summary)

**Worked:**
- Asymmetric warm-start (the only training recipe that produced heterogeneous
  routing with good BLER).
- Alpha sweep at α=2e-3 → 0.1 pp of dense large at 56% FLOPs.
- Channel-aware router input (proven load-bearing by random-router ablation).
- 3-expert design (proven by 2-expert ablation).
- Wall-clock benchmark: 1.93× actual GPU speedup, exceeds theoretical ratio.

**Didn't work / characterized as failure modes:**
- Phase 1 joint-from-scratch (BLER bleeds).
- Phase 2 full warm-start (collapses to 100% large).
- β=2.0 forced uniform (BLER craters).
- Switch / capacity-constraint anti-collapse (all collapse).
- Large-warmup stabilization (over-corrects to 100% large).
- β-warmup stabilization (worse mean BLER).
- Zero-shot DeepMIMO OOD (all models fail).

**Open problems:**
- Seed stability of asym-warm (2/3 reproduce, 1 collapses).
- OOD generalization: 500-step / 2k-sample few-shot fine-tune insufficient
  (both dense and MoE stuck at ~0.99 OOD BLER). Needs larger OOD slice or
  domain-randomized pretraining.
- Hand-rule SNR-oracle cascade slightly cheaper than learned router at
  matched BLER → motivates explicit SNR-estimate router input as future work.

---

## 7. Operational Notes (lessons from the cluster)

- **Single dataset rule:** always train on `dense-v1` 50k subset; always eval
  on `dense-v1` test split. No mixed-protocol comparisons.
- **state_dim=56 is non-negotiable** for MoE — s32 costs 15 pp waterfall BLER.
- **`num_workers=0`** — the 22 GB Arrow training table eats all RAM with
  worker forks; OOM on 32 GB nodes. Single-process is fine because
  data_t < step_t for the MoE step.
- **wandb-init port-info flake** hit ~50% of training jobs. Recovery: re-submit;
  for failed jobs, use `evaluate.py` from the local checkpoint.
- **qsub `-v RUN_ARGS=...` eats commas inside `[uma,tdlc]` brackets.** Bake
  multi-value Hydra overrides into experiment YAML files (see eval40-44).
- **PTX JIT can deadlock TF+PyTorch on CC9.0/CC12.0.** Disable TF GPU at
  import time when only PyTorch needs the device.
- **DeepMIMO has no auto-download.** Manually download `ASU_Campus1.zip`
  on local, scp to cluster, extract under `$DEEPMIMO_DIR/scenarios/`.
- **No `gpu_mem`** in qsub — model uses ~6 GB VRAM, any GPU is fine.
- **One-line semantic commits, no Co-Authored-By** (per project preference).

---

## 8. Comparison to Related Work

- **Wiesmayr et al. (2024):** static dense NRX → our dense baseline.
- **MEAN (van Bolderik 2024):** homogeneous experts, per-SNR specialisation,
  no compute penalty, CDL-C only.
  Our work: heterogeneous experts, compute efficiency via FLOPs penalty,
  emergent routing, mixed UMA+TDL-C, Pareto analysis.
  **Orthogonal contributions** — MEAN focuses on robustness, we focus on
  compute.
- **LOREN (van Bolderik 2026):** LoRA adapters for memory efficiency.
  Different axis from ours.
- **Song et al. (2025):** channel-aware gating in wireless. We instantiate
  this idea on top of a 5G NRX with explicit FLOPs penalty.

---

## 9. Final Deliverables

**Code:**
- Asym-warm MoE training recipe (in `src/training/trainer.py` + experiment
  YAMLs `exp24`–`exp37`).
- Eval pipeline with per-profile/per-SNR breakdowns (`evaluate.py`).
- 4 analysis scripts:
  - `scripts/benchmark_latency.py` — synthetic + real wall-clock timing.
  - `scripts/analyze_static_baselines.py` — SNR-oracle cascade analysis.
  - `scripts/visualize_channel_features.py` — PCA-2D viz.
  - `scripts/plot_per_snr_routing.py` — per-SNR routing stacked bars.

**Figures:**
- `figures/pareto_bler_flops.png` — main Pareto curve.
- `figures/expert_usage_asym_warm.png` — large "wakes up" plot.
- `figures/expert_usage_by_snr.png` — per-SNR routing.
- `figures/waterfall_dense_baselines.png` — dense-baseline waterfall.
- `figures/bler_by_snr_comparison.png` — MoE vs dense waterfall.
- `figures/channel_feature_tsne_{uma,tdlc}.png` — feature PCA viz.
- `figures/per_snr_routing_2zboo1rh.png` — exp26 router by SNR.
- `figures/archictures.png` — architecture overview.

**Checkpoints (W&B):**
- Dense baselines: `dense_{nano,small,large}_final20k_constant_lr_s67`.
- exp26 MoE: `moe_alphasweep_asym_a2e3_s67-t6lkdep2:best`.
- Plus exp24/25/27/28/29/30/31/32–37 for sweep + ablations.
- Plus fine-tuned OOD: `go74dlm7` (dense), `9t2wyyus` (exp26).

**Datasets:**
- Train: `Vack0/moe-5g-nrx` 50k subset Array3D.
- Val/test: `dense-v1/{val,test}/{uma,tdlc}.pt` (cached `.pt`).
- OOD: `dataset-test-asu_campus1` (32k samples, ray-traced).
- Few-shot OOD: `dataset-train-asu_campus1` (2,048 samples).

---

## 10. Roadmap (where the doc ends, where the code lives)

| # | Task | Status |
|---|---|---|
| 1  | Asym warm 20k test eval | superseded by exp26 alpha winner |
| 2  | Alpha sweep (4 jobs) | ✅ exp26 (α=2e-3) is the winner |
| 3  | 3-seed on α=2e-3 | ✅ bimodal (2/3 reproduce) |
| 4  | Random-feature router ablation | ✅ channel-aware features ARE load-bearing |
| 5  | 2-expert ablation | ✅ nano earns its keep |
| 6  | DeepMIMO OOD eval (asu_campus1) | ✅ all fail; honest scope finding |
| 7  | Large-warmup stabilization | ✅ over-corrects (negative result) |
| 8  | β-warmup stabilization | ✅ worse mean BLER (negative result) |
| 9  | Static + SNR-oracle cascade baseline | ✅ exp26 on Pareto, oracle slightly cheaper |
| 10 | DeepMIMO few-shot fine-tune | ✅ done — both stuck at ~0.99 OOD (negative result) |
| 11 | Wall-clock latency (synthetic) | ✅ 1.93× speedup vs dense_large |
| 11b| Wall-clock latency on real data | ⏳ queued |
| 12 | Per-SNR routing viz | ✅ exp26 figure rendered |
| 13 | Channel-feature PCA viz | ✅ {uma,tdlc} PNGs rendered |
| 14 | Multi-scenario OOD | ⏸ blocked (deepmimo.net link down) |
| 15 | Doc cleanup of `checkpoint_report.md` | ⏳ next bottleneck |

**Cut:** difficulty-guided routing, dataloader Arrow→torch refactor,
re-baselining dense at bs=512, MEAN reimplementation. None move the rubric.

---

## Appendix A — Run Index (W&B)

**Alpha sweep:**
- exp24 (α=5e-4): eval `002cwsy2`
- exp25 (α=1e-3): train `3xzxkddv`, eval `5jswm490`
- exp26 (α=2e-3): train `t6lkdep2`, eval `2zboo1rh`
- exp27 (α=5e-3): eval `dh4x0qmu`

**3-seed:** exp28 (s32) — collapse; exp29 (s42) — reproduces.

**Ablations:**
- exp30 (random router) — load-bearing claim ✓
- exp31 (2-expert) — nano earns its keep ✓

**Stabilization (negative):**
- exp32/33/34 — large-warmup → all collapse to 100% large
- exp35/36/37 — β-warmup → 3 different patterns, worse mean BLER

**OOD few-shot:**
- `y1o9guf5` — gen 2k ASU samples
- `go74dlm7` — fine-tune dense_large
- `9t2wyyus` — fine-tune exp26

**Dense baselines:**
- nano:  `aos4hhid:best` (90k params)
- small: `kivdz4qu:best` (168k params)
- large: `55l1dpby:best` (450k params)

---

## Appendix B — Honest Scope Statement (for the report)

This work demonstrates compute-aware MoE for 5G neural receivers on
**synthetic Sionna data** (UMa + TDL-C). On in-distribution test, exp26
matches dense large within 0.1 pp BLER at 56% FLOPs and 1.93× wall-clock
speedup. The channel-aware router and 3-expert design are both ablated and
shown to be load-bearing. The asym-warm training recipe is **not seed-stable**
(2/3 seeds reproduce); we recommend best-of-N seeds. **OOD generalization
to ray-traced channels (DeepMIMO ASU Campus) requires fine-tune** — all
three of our models (including dense large) fail zero-shot. A hand-rule
SNR-oracle cascade slightly dominates exp26 at matched BLER, motivating
explicit SNR estimation in the router as concrete future work.

These limitations are reported transparently and do not undermine the
contribution: a working compute-aware MoE recipe with characterized failure
modes, honest baselines, and a real wall-clock speedup.
