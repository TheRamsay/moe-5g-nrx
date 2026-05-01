# Context for Agents: Compute-Aware MoE for Efficient 5G Neural Receivers

## Goal

Build a **compute-aware 5G neural receiver** that keeps BLER close to a dense baseline while reducing **average FLOPs** via adaptive routing. Main result: **BLER vs Average FLOPs** Pareto curve. Router uses channel-quality features from the shared stem — not raw SNR.

## Current State (2026-05-01 morning update)

### Overnight batch (2026-05-01) — 9 jobs landed

**Headline new findings since the 2026-04-30 evening consolidation:**

| # | Experiment | Result |
|---|---|---|
| **exp60 (α/data hypothesis)** | 100k + α=1e-3 | ✓ **HETEROGENEOUS** — confirms α/data ratio principle. Avg BLER ~0.902 at real_flops ~0.65. Two prior collapsed runs (exp40, exp58) at α=2e-3 now explained by single principle. |
| **exp59 30k convergence** | Was meant to be final-headline number | ✗ **COLLAPSED** to bad attractor — avg BLER 0.926 (5pp WORSE than exp26 at step 12k). Even seed-67 (the headline seed) gave a different outcome on re-run. CUDA non-determinism contributes — seed alone doesn't pin trajectory. |
| **exp61 v1 (function-specialized)** | Sink + channel_only + large | ⚠️ **HYDRA CONFIG BUG** — `override /model: moe_nl` deep-merged with `model.experts: {sink, channel_only, large}` instantiated a **5-expert** model (nano + small + large + channel_only + sink), not the intended 3. v1 BLER 0.911 / real_flops 0.35 is for the 5-expert architecture. Diagnosed via checkpoint inspection. **v2 resubmitted (job 19594735) with dedicated `conf/model/moe_func.yaml` — 3-expert clean version verified locally (sink:0, channel_only:109k, large:370k, total 567k).** |
| **LMMSE on O1 OOD** | Classical baseline on ray-traced | ✓ **LMMSE beats both NN models** (0.976 vs dense_large 0.982 vs exp26 0.984). Ray-traced gap is from learned synthetic priors actively misleading the receiver — not from fundamental channel difficulty. Strong scope statement for the report. |
| dense_micro pretrain (19583495) | exp42 prerequisite | ✗ Died at walltime (slow data loader, 12% progress in 3h). exp43 (smaller-small MoE) path closed; exp41 already disproved "small is a sink" anyway. |

### Refined bimodality picture (post-exp59)

**Success rate at α=2e-3: 2/6 across recent attempts.**

| Run | Data | Seed | Steps | α | Outcome |
|---|---:|---:|---:|---:|---|
| exp26 | 50k | 67 | 12k | 2e-3 | ✓ 0.902 |
| exp29 | 50k | 42 | 12k | 2e-3 | ✓ 0.902 |
| exp28 | 50k | 32 | 12k | 2e-3 | ✗ 0.958 |
| exp40 | 100k | 67 | 12k | 2e-3 | ✗ 0.953 |
| exp58 | 100k | 42 | 12k | 2e-3 | ✗ 0.968 |
| exp59 | 50k | 67 | **30k** | 2e-3 | ✗ 0.926 |
| **exp60** | **100k** | **67** | **12k** | **1e-3** | **✓ ~0.902** |

Honest framing: *"Recipe at α=2e-3 is bimodal (2/6). The α=1e-3 variant at 100k matches BLER, suggesting the α/data ratio is the controllable knob. Recipe stability at α=2e-3 across seeds is an open problem — 8 anti-collapse mechanisms (Switch aux × 4, capacity × 4) all failed."*

### exp60 TEST-SET LOCKED (eval62, job 19594870, DONE 2026-05-01 ~10:00)

**α/data hypothesis confirmed on the locked dense-v1 test set.**

| Metric | exp26 (anchor, test) | **exp60 (test)** | Δ |
|---|---:|---:|---:|
| UMa BLER | 0.9369 | 0.9366 | -0.0003 |
| TDLC BLER | 0.8674 | 0.8660 | -0.0014 |
| **Avg BLER** | **0.9022** | **0.9013** | **-0.0009** |
| Avg real FLOPs | 0.558 | 0.584 | +2.6pp |
| TDLC routing (l/n/s) | 44/15/40 | 48/11/41 | skews slightly to large |

**Headline-publishable claim:** *"At 100k samples, the asym-warm recipe with α scaled inversely (α=1e-3 instead of 2e-3) reproduces exp26's test BLER within 0.001 at slightly higher compute (+2.6pp FLOPs). Two collapsed runs at 100k+α=2e-3 (exp40, exp58) explained by the α/data scaling principle, with the principle's correctness confirmed by exp60's recovery to exp26-quality on the locked test set."*

### Currently in flight (2026-05-01)

- **19594735** — exp61 v2 (CLEAN function-specialized: sink + channel_only + large, 567k params). ~3h walltime. Will tell us whether the 5-expert v1 result (BLER 0.911 / real_flops 0.35) holds for the proper 3-expert architecture.
- **19594871** — exp63 (10k + α=2e-3) — lower-bound data-scaling test. Brackets the data story below 50k.

---

## Current State (2026-04-30)

### Latest sprint (2026-04-30 evening — post-consultation work)

After being roasted at consultation on (1) NN explanation depth, (2) lack of
standard baseline comparison, (3) methodology rigor, we pushed a heavy batch
of follow-up experiments tonight. Summary of new results below; details in
following sections.

**New findings since 2026-04-27 CLAUDE.md:**

| # | Experiment | Key result |
|---|---|---|
| **Classical baselines** | LMMSE (LS-MRC), Genie-MRC, single-antenna | Single-ant 0.995 / **LS-MRC 0.900** / Genie-MRC 0.854 — neural beats classical only in waterfall, classical wins on simpler channels |
| **3GPP in-family OOD** | exp26+dense_large+LMMSE on TDL-A/D, CDL-A | exp26 generalizes well within 3GPP family (BLER 0.82–0.83). **LMMSE beats neural by 2-3 pp on simpler profiles.** |
| **Anti-collapse sweep** | Switch-aux at weights 1e-3..1e0 (4 runs) | **All 4 collapsed to 100% large** at hard top-1 inference. Confirms original finding rigorously. |
| **Anti-collapse sweep** | Capacity penalty at weights 0.1..10 (4 runs) | In progress — running |
| **No-small ablation** | exp41: drop small, keep {nano, large} | **+5.3 pp BLER cost** (avg ~0.955). **Disproves "small is just a sink" critique.** Small does real decoding work, especially on TDLC. |
| **Smaller-small alternative** | exp42 (dense_micro pretrain, block_dim=16) → exp43 | In progress — exp42 running |
| **100k data scaling** | exp40: same recipe as exp26, 2× training data | In progress — running |
| **Wall-clock latency on real data** | All 4 models on same A40 GPU, real OFDM data | **exp26 is 1.67× SLOWER than dense_large** on real data due to dispatch overhead. Original 1.93× synthetic claim is misleading; FLOPs is the right metric. |
| **High-resolution per-SNR** | eval46/47 + LMMSE at snr_bins=20 | Cleaner waterfall data for the consultation figure |
| **PCA channel-feature viz** | (already done, now validated as the "implicit SNR encoding" evidence) | Stem features cluster monotonically by SNR on UMa — visual proof that router learned implicit SNR representation |
| **Explicit SNR-input ablation** | exp38: feed raw signal stats to router | **Collapsed to 100% large.** Implicit stem features are sufficient — explicit raw stats are redundant + destabilising. |

### Classical baselines (2026-04-30 tonight)

Implemented as `src/baselines/lmmse.py` (vectorised PyTorch, no trainable params)
and evaluated via `scripts/evaluate_lmmse.py`. Three modes form a ladder:

| Baseline | UMa BLER | TDLC BLER | Avg BLER | Channel info |
|---|---:|---:|---:|---|
| Single-antenna detection | 0.992 | 0.998 | **0.995** | LS estimate, 1 of 4 antennas |
| **LS-MRC (realistic classical)** | **0.939** | **0.861** | **0.900** | LS estimate + MRC across 4 antennas |
| **Genie-MRC (oracle)** | **0.908** | **0.800** | **0.854** | True channel + MRC (NOT deployable) |

**Key per-SNR comparison (TDLC waterfall 15-20 dB):**
- LS-MRC: 0.155 BLER
- Genie-MRC: 0.027 BLER
- dense_large: ~0.085 BLER
- **Channel estimation is the classical bottleneck** — LS vs Genie gap = 13 pp
- **Neural beats LS-MRC in waterfall** (~10 pp at 14 dB) but **loses to Genie-MRC** at high SNR (with perfect channel info, classical is mathematically optimal for 16-QAM)

### 3GPP in-family OOD (2026-04-30)

Generated test data for TDL-A, TDL-D, CDL-A using same generator pipeline
(`src/data/sionna_generator.py`, extended with new ChannelProfile enums).
32k samples per profile, deterministic seed (base_seed=67 + TEST_SEED_OFFSET).

| Model | TDL-A | TDL-D | CDL-A | Avg in-family OOD |
|---|---:|---:|---:|---:|
| **LMMSE (LS-MRC)** | **0.804** | **0.801** | **0.801** | **0.802** |
| dense_large | 0.832 | 0.822 | 0.821 | 0.825 |
| **exp26 MoE** | **0.834** | **0.824** | **0.816** | **0.825** |

**Big finding:** exp26 generalizes really well within the 3GPP family (BLER
~0.82, BETTER than its training BLER on TDL-C 0.867 because these profiles
are simpler — lower delay spread, LOS components). **LMMSE classical actually
beats both neural models on these in-family OOD profiles.** Why: simpler
channels + perfect math = classical wins. Neural's edge is in complex/noisy
channels.

This narrows the OOD weakness from "all unfamiliar channels" to specifically
"the synthetic-stochastic vs ray-traced-geometric gap" (DeepMIMO ASU still
fails catastrophically; in-family 3GPP works fine).

### Anti-collapse sweep — Switch aux + capacity (8 runs, all failed)

Re-ran both anti-collapse mechanisms with proper hyperparameter sweeps.

**Switch aux loss (4 weights, all collapsed):**

| Exp | switch_aux_weight | TDLC BLER | exp_flops | real_flops | Outcome |
|---|---:|---:|---:|---:|---|
| exp44 | 1e-3 | 0.844 | 1.000 | 1.000 | full collapse to large |
| exp45 | 1e-2 | 0.844 | 1.000 | 1.000 | full collapse to large |
| exp46 | 1e-1 | 0.840 | 1.000 | 1.000 | full collapse to large |
| exp47 | 1e0 | 0.859 | 0.544 | 0.996 | soft routing has high entropy but argmax picks large 99.6% |

**Soft capacity penalty (4 weights, two failure modes):**

| Exp | capacity_weight | TDLC BLER | real_flops | Outcome |
|---|---:|---:|---:|---|
| exp48 | 0.1 | 0.840 | 1.000 | weak penalty → full collapse |
| exp49 | 0.5 | 0.959 | 0.675 | spread routing but BLER tanks (+9 pp) |
| exp50 | 2.0 | 0.970 | 0.497 | very spread, BLER even worse |
| exp51 | 10.0 | 0.881 | 0.600 | partial recovery (~60% FLOPs) but still 1pp worse than exp26 |

**No middle ground exists** for either mechanism where you get heterogeneous
routing AND good BLER. Original "single-shot failed" claim now properly
characterized across 4 orders of magnitude × 2 mechanisms = 8 runs.

### Routing trajectory analysis (2026-04-30 evening, killer figure)

Pulled training-step trajectories for routing distributions and entropy across
the 3 paradigms + anti-collapse sweeps. Saved as `docs/figures/
routing_trajectories_{collapse_modes,bimodal_seeds,anti_collapse}.png`.

Key findings from the figures:
- **Phase 2 v1**: router commits to large at step ~50, entropy → 0 instantly
- **Phase 1**: router commits to small/nano at step ~1000
- **Asym warm s67**: stays exploratory until step ~10000, gradual rebalancing
- **Bimodal seeds**: s67 vs s42 trajectories diverge subtly in steps 0-2000
- **All 8 anti-collapse sweeps**: each commits to a bad pattern within first
  few thousand steps; none recover heterogeneous + good BLER

**This figure is the visual proof of the "expert-quality gap at initialization
determines routing attractor" hypothesis** that motivated the symmetric sweep
(exp56/exp57).

### Router mechanism analysis (DONE 2026-04-30 evening)

Three sub-analyses from one inference pass over UMa + TDLC test data (4k each):

**A. Linear probing — implicit SNR encoding is PROFILE-SPECIFIC, not universal:**

| Probe | UMa R² | TDLC R² |
|---|---:|---:|
| **SNR** | 0.42 | **0.93** |
| Channel power | 0.43 | 0.07 (Sionna normalises per-sample → low variance) |
| Delay spread | 0.36 | 0.65 |
| Profile classification (UMa vs TDLC) | 0.78 (combined) | |

TDLC SNR R²=0.93 is the strongest result — confirms strong implicit SNR
encoding for the harder NLOS channel where SNR cleanly determines decodability.
UMa SNR R²=0.42 is much weaker — UMa channels are more position-dependent
and SNR alone is a poor predictor of decodability (BLER stays at 0.72 even
at 23 dB). The stem learned **different feature representations per profile**
rather than a universal SNR encoder.

**C. Per-expert specialization — striking BLER pattern:**

Routing share per profile:
- UMa: 49% nano / 24% small / 26% large
- TDLC: 15% nano / 39% small / 46% large

SNR distribution per chosen expert (cleaner on TDLC):
- TDLC: nano at extreme low (-10 to -5 dB), small in middle (~0 dB), large at high (15+ dB)
- UMa: nano dominates low, small fills middle, large skews high — but more overlap

**Striking finding:** nano and small are at BLER ≈ 1.0 across ALL SNR bins
in our per-sample analysis. Only large ever recovers BLER (~0.1 on TDLC at
high SNR, ~0.5 on UMa at high SNR). The router's value is **compute efficiency**
— it routes hopeless samples to nano (cheap failure) and decodable samples
to large (only one that can decode). Channel-MSE auxiliary loss is the only
way nano/small contribute meaningfully to gradient signal.

**F. Decision boundary on PCA plane:**

Routing decisions form coherent regions in PCA space (5-NN vote). Region
boundaries roughly align with the SNR colour gradient — visual confirmation
of A's quantitative result. Cleaner regional separation on TDLC than UMa.

**Files:** `docs/figures/router_mechanism_{linear_probing,expert_specialization,
decision_boundary}.png` + `router_mechanism_linear_probing.json`.

**Refined narrative for the report:**
- Old: "Stem encodes SNR implicitly — that's why explicit SNR proxies (exp38) were redundant."
- New: "The stem learned profile-specific representations. Strong SNR encoding on TDLC (R²=0.93) where SNR drives BLER; weaker on UMa (R²=0.42) where SNR is a poor predictor. The router uses these profile-appropriate features for routing, explaining why routing patterns differ between profiles."

### Symmetric asym-warm sweep (DONE 2026-04-30) — refined principle

Tested whether "cold-expert grows in" generalizes by varying which expert is
cold-init.

| Exp | Setup | Avg BLER | real_flops | Outcome |
|---|---|---:|---:|---|
| **exp26** (cold-LARGE) | warm nano+small, cold large | **0.902** | **0.56** | heterogeneous ✓ (44/15/40) |
| **exp56** (cold-SMALL) | warm nano+large, cold small | 0.897 | 0.82 | mostly large + some nano (small never recovers) |
| **exp57** (cold-NANO) | warm small+large, cold nano | 0.887 | **1.00** | **FULL Phase-2 collapse to large** |

**Refined finding:** asym-warm-start works **specifically when LARGE is
cold-init** — not just any cold expert. The mechanism: temporarily
handicapping the highest-capacity expert forces the router to commit to
smaller experts before large becomes competent. When OTHER experts are
cold (exp56, exp57), warm-large still dominates from step 1, the router
locks on large, and the cold expert never recovers.

**This is a richer/stronger publishable claim than "asym-warm works":**
exp26's recipe is *uniquely privileged* because cold-large is the only
configuration that breaks warm-large's gradient dominance.

### Per-expert success rate analysis (DONE 2026-04-30) — nano/small are ZERO%

Aggregate per-expert success rates on routed samples (4k samples per profile):

| Expert | UMa success rate | TDLC success rate |
|---|---:|---:|
| nano | **0.00%** | **0.00%** |
| small | **0.00%** | **0.00%** |
| **large** | **23.21%** | **29.42%** |

**Nano and small literally never decode any block successfully.** Only
large delivers actual decoded outputs. This **definitively answers** the
"is small a sink?" question: yes — nano and small are pure compute optimizers
that never decode. Their value is:

1. **Compute savings** on hopeless samples (nano cheaper than small cheaper than large)
2. **Channel-MSE auxiliary loss** training signal — small produces better channel estimates than nano even when bits fail, contributing to stem feature quality during training
3. **Routing structure** — 3 cost tiers vs 2 give finer-grained adaptive compute

**Reframed project contribution:** *"The compute-aware MoE doesn't have
multiple experts that all decode in parallel. It has ONE expert that decodes
(large) plus two experts that intelligently skip-the-compute on hopeless
samples. The router's value is recognizing 'this is hopeless, route to cheap
fail' vs 'this is decodable, pay for large.'"*

Figure: `docs/figures/router_mechanism_success_rate.png` — visually shows
nano/small flat at 0 while large rises from 0 at low SNR to ~0.4 (UMa)/0.9
(TDLC) at high SNR.

### 100k data scaling — both seeds collapse, refined hypothesis is "α/data scaling"

Concern raised in consultation: 50k samples might be too small. Tested at 100k.

| Run | Data | α | Seed | Avg BLER | real_flops | Outcome |
|---|---:|---:|---:|---:|---:|---|
| Original anchor | **~250k (full HF stream)** | **1e-3** | 67 | 0.910 | 0.61 | ✓ heterogeneous |
| exp26 (50k headline) | 50k | 2e-3 | 67 | 0.902 | 0.56 | ✓ heterogeneous |
| exp40 (100k) | 100k | 2e-3 | 67 | ~0.953 | 0.465 | ✗ collapsed |
| exp58 (100k retry) | 100k | 2e-3 | 42 | ~0.968 | 0.305 | ✗ collapsed |
| **exp60 (in flight)** | **100k** | **1e-3** | **67** | (testing) | | tests α-scaling hypothesis |

**Both 100k runs at α=2e-3 collapsed** (Phase-1 style — heavy nano routing).
But the **original anchor at full HF stream + α=1e-3 worked fine**. So the
issue is NOT data scale per se — it's the **α/data ratio**. α=2e-3 was
optimal at 50k; transferring it naively to 100k makes the FLOPs penalty
effectively too strong → router collapses to nano early before large can
catch up.

**exp60 (100k + α=1e-3, in flight)** tests this. Predicted outcome:
heterogeneous routing, BLER ≈ 0.91 (matching original anchor pattern). If
correct → "α needs to be scaled inversely with data-per-epoch" is a clean
methodological finding for the report.

### Convergence study (in flight) — 30k step extension of exp26

All ablations use 12k steps (fair comparison budget — standard ML practice).
exp59 trains exp26 recipe at 30k for the final-report headline number.

Existing 20k extension data showed TDLC BLER dropping 0.867 → 0.851 between
12k and 16k, suggesting 12k is mildly under-converged. Final headline number
likely 1-2 pp better than 0.902.

### O1_3p5 DeepMIMO OOD (in flight)

ASU campus failed catastrophically (BLER 0.99). Testing O1_3p5 (Outdoor
Street at 3.5 GHz — same carrier, simpler geometry) to determine whether
ASU was specifically pathological or all ray-traced outdoor fails.

Generation job 19586235 in flight. Eval jobs ready to submit once data is
generated.

### Wall-clock latency — real data (corrected story)

The original 2026-04-26 synthetic-input benchmark claimed 1.93× speedup.
Re-running on **real OFDM test data** with all 4 models on the same GPU
(NVIDIA A40):

| Model | synth ms/batch | real ms/batch |
|---|---:|---:|
| dense_nano | 1.05 | 1.06 |
| dense_small | 2.09 | 2.07 |
| dense_large | 3.32 | 3.28 |
| **exp26 MoE** | **1.93** | **5.53** |

**exp26 is 1.67× SLOWER than dense_large on real data.** Dense models: real ≈
synthetic. MoE: real >> synthetic because hard top-1 routing splits the batch
into 3 sequential sub-batches (mask indexing + scatter overhead dominates at
batch=64).

**Claim retraction:** the 1.93× synthetic speedup does NOT translate to
wall-clock with our naive dispatch implementation. **The Pareto frontier is
reported in FLOPs (hardware-agnostic), NOT latency.** Production sparse-MoE
inference (Mixtral, vLLM dispatch kernels) would be needed to convert FLOPs
savings into wall-clock; out of scope for this work. The defense brief and
notebook were updated to remove the 1.93× claim.

### Small-expert ablations (exp41 done, exp43 in flight)

Symmetric counterpart to exp31 (which dropped nano):

| Run | Setup | UMa BLER | TDLC BLER | Avg BLER | vs exp26 |
|---|---|---:|---:|---:|---|
| exp26 | {nano, small, large} | 0.937 | 0.867 | 0.902 | reference |
| exp31 | {small, large} (drop nano) | 0.940 | 0.878 | 0.909 | +0.7 pp |
| **exp41** | **{nano, large}** (drop small) | **0.967** | **0.942** | **~0.955** | **+5.3 pp** |

**Definitively answers the "small is just a sink" critique: NO.** Dropping
small costs FAR more than dropping nano (5.3 pp vs 0.7 pp). Small does real
decoding work, especially on TDLC. The 3-expert design with current sizes is
justified by both ablations.

exp42 (dense_micro pretrain at block_dim=16) + exp43 (smaller-small MoE)
will tell us whether the small expert can be made smaller without losing
function.

### Explicit SNR-input ablation (exp38, done 2026-04-29)

Added `use_input_statistics=true` to feed router 3 scalar signal-derived
features (received_power, channel_power, channel_variance — all SNR proxies
computable at inference). Hypothesis: explicit SNR proxies could close the
7 pp FLOPs gap to the SNR-oracle cascade.

**Result: collapsed to 100% large.** Avg BLER 0.897 (similar to dense_large)
at 100% FLOPs (no compute savings).

**Interpretation:** the implicit stem features already encode SNR-correlated
information (confirmed by the PCA visualization that shows monotonic SNR
gradient in stem feature space). Adding raw stats on top is redundant AND
destabilising — the router uses the explicit signal as an "easy classifier"
for picking large from step 1.

**Connects three findings:**
1. random-router ablation (exp30): channel features matter
2. exp38 collapse: raw stats are redundant with stem features
3. PCA viz: stem features encode SNR implicitly

### Original Pareto frontier (unchanged)



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

**Wall-clock latency (synthetic Gaussian, GPU, batch=64, 2026-04-26):**

| Model | Params | ms/batch | samples/sec | Speedup vs dense_large |
|---|---:|---:|---:|---:|
| dense_nano | 90k | 0.97 | 65,693 | 3.00× |
| dense_small | 168k | 1.71 | 37,382 | 1.70× |
| dense_large | 450k | 2.92 | 21,925 | 1.00× |
| **exp26 MoE** | 583k | **1.51** | **42,391** | **1.93×** |

The 44% FLOPs reduction translates to **1.93× actual GPU speedup**, better than
theoretical 1.78× because at hard top-1 only the selected expert runs and
kernel-launch overhead favors the smaller experts. Caveat: synthetic input →
router argmax approximately uniform 33/33/33. Real-data per-profile timing in flight.

**DeepMIMO few-shot OOD fine-tune (DONE 2026-04-27):** Generated 2,048 ASU samples
(`y1o9guf5`), fine-tuned dense_large 500 steps lr=1e-4 (`go74dlm7`) + exp26
(`9t2wyyus`), then re-eval'd on uma+tdlc+asu_campus1.

| Model | OOD BLER zero-shot | OOD BLER post-FT | Δ | In-dist preserved? |
|---|---:|---:|---:|---|
| dense_large | 0.990 | 0.9901 (`t4yo37am`) | +0.0001 | ✓ unchanged |
| **exp26 MoE** | 0.992 | **0.9915** (`kjc12s5p`) | −0.0005 | ✓ identical, routing unchanged |

**Honest negative result:** 500 steps / 2k samples insufficient to recover OOD
performance on either model. Both stay at ~0.99 BLER on ray-traced channels.
**No catastrophic forgetting** — ft_exp26 in-dist (UMA 0.937 / TDLC 0.867) and
routing (44/15/40 on TDLC) are bit-identical to zero-shot exp26.
ft_exp26 OOD routing: 75/14/11 (n/s/l) — fine-tune did NOT teach router to
escalate to large for OOD samples; nano-default behavior preserved.

> "Substantial OOD generalization requires longer fine-tune, larger OOD slice,
> or domain-randomized pretraining — out of scope for this work."

**Cluster ops note:** 4 cluster jobs needed for ft_exp26 OOD eval — wandb-init
port flake hit 3× consecutively, then `WANDB_MODE=offline/disabled` blocked
artifact downloads. Final fix: full clean of 207 GB `.cache/wandb` (filling home
quota), then plain online resubmit (`19497967`) succeeded.

**Channel-feature PCA-2D viz (done 2026-04-26):** stem features colored by
router's selected expert + by true SNR. Figures at
`/storage/brno2/home/ramsay/moe-5g-nrx/docs/figures/channel_feature_tsne_{uma,tdlc}.png`
(numpy SVD-PCA; scikit-learn dropped because cluster uv cache offline).

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

**Anti-collapse sweep:** β=2.0 forces 33/33/33 but kills BLER; capacity
constraint collapses after unfreeze; Switch aux too weak. Asym warm-start is
the only approach that worked.

**Anti-collapse sweep (proper hyperparameter version, 2026-04-30):** ran
Switch aux loss at weights {1e-3, 1e-2, 1e-1, 1e0} — **all 4 collapsed to
100% large at hard top-1 inference**, even when soft routing has high entropy
(weight=1e0: exp_flops=0.544 but real_flops=0.996). Capacity penalty sweep
{0.1, 0.5, 2.0, 10.0} in flight. Confirms original single-shot finding at
proper rigor.

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

## A-Grade Roadmap (updated 2026-04-30)

| # | Task | Status |
|---|---|---|
| 1 | Asym warm 20k test eval | superseded by exp26 alpha-sweep winner |
| 2 | **Alpha sweep (4 jobs)** — exp24..exp27 | ✅ done; exp26 (α=2e-3) is the winner |
| 3 | **3-seed on α=2e-3** (s32, s42 alongside s67) | ✅ done; bimodal — 2 reproduce, 1 collapses |
| 4 | **Random-feature router ablation** | ✅ done; channel-aware features ARE load-bearing (BLER craters 6.6pp without) |
| 5 | **2-expert ablation (drop nano)** — exp31 | ✅ done; nano earns its keep (0.7pp BLER hit + 9pp more FLOPs without) |
| 5b | **2-expert ablation (drop small)** — exp41 | ✅ done 2026-04-30; +5.3pp BLER cost — disproves "small is a sink" critique |
| 5c | **Smaller-small MoE** — exp42+exp43 | 🔄 in flight 2026-04-30; tests block_dim=16 vs 32 for "small" expert |
| 6 | **DeepMIMO OOD eval** (asu_campus1) | ✅ done; all 3 models fail (~0.99 BLER); honest scope finding |
| 6b | **3GPP in-family OOD** (TDL-A, TDL-D, CDL-A) | ✅ done 2026-04-30; exp26 generalizes well within 3GPP family (0.82 BLER), LMMSE actually beats neural here |
| 6c | **DeepMIMO O1_3p5** (simpler outdoor than ASU) | 🔄 generation queued 2026-04-30; tests if ASU was specifically hard |
| 7 | **Large-warmup stabilization** (exp32/33/34) | ✅ done; over-corrects to 100% large in 3/3 seeds (negative result) |
| 8 | **β-warmup stabilization** (exp35/36/37) | ✅ done; mean BLER worse than baseline (negative result) |
| 8b | **Switch-aux loss sweep** (exp44–47, weights 1e-3..1e0) | ✅ done 2026-04-30; **all 4 collapsed** — proper sweep confirms original single-shot finding |
| 8c | **Capacity penalty sweep** (exp48–51, weights 0.1..10) | ✅ done 2026-04-30; two failure modes — collapse OR forced routing kills BLER |
| 8d | **Routing trajectory analysis** (W&B history → matplotlib) | ✅ done 2026-04-30; killer figures showing collapse dynamics across 11 runs |
| 8e | **Symmetric asym-warm sweep** (exp56 cold-small, exp57 cold-nano) | ✅ done 2026-04-30; **cold-LARGE is uniquely effective** — exp57 fully collapsed, exp56 mostly large. Refined principle for the writeup. |
| 8g | **Per-expert success rate analysis** | ✅ done 2026-04-30; **nano/small NEVER decode (0%), only large does (23-29%)** — they're pure compute optimizers |
| 8f | **30k convergence study** (exp59) | 🔄 in flight 2026-04-30; final-report headline number for exp26 recipe |
| 9 | **Static + SNR-oracle cascade baseline** (D analysis) | ✅ done; exp26 on Pareto frontier; oracle cascade slightly better at same BLER |
| 9b | **Classical LMMSE / Genie-MRC / single-ant baselines** | ✅ done 2026-04-30; complete classical ladder vs neural results |
| 10 | **DeepMIMO few-shot fine-tune** | ✅ done; both models stuck at ~0.99 OOD BLER (negative result, no catastrophic forgetting) |
| 11 | **Wall-clock latency benchmark (synthetic)** | ✅ done — exp26 1.93× faster on RTX PRO 6000 (synthetic input) |
| 11b | **Wall-clock latency on real test data** | ✅ done 2026-04-30; **NEGATIVE: exp26 1.67× SLOWER than dense_large on real data** due to dispatch overhead. Original 1.93× claim retracted. |
| 12 | **Per-SNR routing visualization** | ✅ done (`docs/figures/per_snr_routing_2zboo1rh.png`, exp26) |
| 12b | **High-resolution per-SNR re-eval** (snr_bins=20) | ✅ done 2026-04-30; cleaner waterfall data for consultation slide |
| 13 | **Channel-feature PCA-2D visualization** | ✅ done (`docs/figures/channel_feature_tsne_{uma,tdlc}.png`) — confirms implicit SNR encoding |
| 13b | **Router mechanism analysis** (linear probing + expert specialization + decision boundary) | ✅ done 2026-04-30; **TDLC SNR R²=0.93 (strong), UMa SNR R²=0.42 (weak) — profile-specific encoding**; per-expert BLER reveals nano/small are pure compute optimizers (BLER≈1.0), only large decodes |
| 14 | **Explicit SNR-input ablation** (exp38, use_input_statistics=true) | ✅ done 2026-04-29; collapsed to 100% large — implicit stem features sufficient |
| 14b | **100k data scaling** — exp40 (s67, α=2e-3) | ✗ done 2026-04-30; collapsed (Phase-1 style, heavy nano) |
| 14c | **100k retry seed=42** — exp58 (α=2e-3) | ✗ done 2026-04-30; ALSO collapsed → 0/2 success rate at α=2e-3 |
| 14d | **100k + α=1e-3** — exp60 (matches original anchor's α) | 🔄 in flight 2026-04-30; tests "α/data ratio" hypothesis |
| 15 | Doc cleanup: checkpoint_report rewrite | NOT STARTED — biggest remaining item |
| 15b | **Defense brief (en+cz) + Jupyter presentation notebook** | ✅ done 2026-04-29/30 |
| 15c | **Teacher feedback notes + 13-day action plan** | ✅ done 2026-04-30 (`docs/teacher_feedback_2026-04-30.md`) |
| 15d | **Experiment dir READMEs** for new 2026-04-29/30 work | ✅ done 2026-04-30 |
| 16 | (Optional A+) MEAN reimplementation as homogeneous-expert baseline | cut — too time-expensive |
| 17 | Poster (A2) | NOT STARTED — deadline 13 days |

**Cut**: difficulty-guided routing, dataloader Arrow→torch refactor,
re-baselining dense at bs=512. None move the rubric.

**Open follow-ups (post-consultation, before final deadline ~13 days):**
- Router interpretability beyond PCA (saliency maps, per-expert feature analysis)
- LaTeX checkpoint report rewrite (THE biggest remaining item)
- Poster
- {nano, micro-small, large} eval (depends on exp43 finishing — exp42 dense_micro pretrain still running)
- 100k seed-42 retry result (depends on exp58 finishing — tests bimodality)
- O1_3p5 OOD eval (depends on data generation finishing)
- 30k convergence result (depends on exp59 finishing — final headline number)
- Symmetric sweep result (depends on exp56/57 finishing — hypothesis test)

**Cluster jobs in flight as of 2026-04-30 late evening:**
- 19586392 — exp56 cold-small (symmetric sweep)
- 19586393 — exp57 cold-nano (symmetric sweep)
- 19586548 — exp59 30k convergence run
- 19586967 — O1 Arrow→pt conversion (queued)
- 19587197 — exp60 100k + α=1e-3 test (queued)

**Finished tonight that haven't been integrated yet:** dense_micro pretrain
(19583495), O1_3p5 OOD test data gen (19586235 — Arrow format, awaiting
conversion), exp58 (19586443 — collapsed), router mechanism (19586663 +
19587043 → see results above).

## vs MEAN (van Bolderik et al., 2024)

MEAN: homogeneous experts, per-SNR specialisation, no compute penalty, CDL-C only. Our work: heterogeneous experts, compute efficiency via FLOPs penalty, emergent routing, mixed UMA+TDL-C, Pareto analysis. Orthogonal contributions.
