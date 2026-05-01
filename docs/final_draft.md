# Compute-Aware MoE as a Training Scaffold for Efficient 5G Neural Receivers

**Team:** Dominik Huml (xhumld00), Jakub Kontrik (xkontr02), Martin Vaculik (xvaculm00)
**Date:** 2026-05-01

---

## TL;DR

We built a **compute-aware 5G neural receiver** with a heterogeneous Mixture-of-Experts (MoE).
On in-distribution Sionna UMa + TDL-C test data, **exp26** matches the dense-large baseline within
**0.1 pp BLER at 56% FLOPs**. We then show — via per-expert success-rate analysis and
counterfactual routing — that the middle "small" expert **never decodes a block** but produces
77%-correct partial bits. From this we derive **Mode B inference**: zero-out small's expert at
deployment time. Result: **avg BLER 0.9021 at 47% FLOPs with zero retraining** — a clean Pareto
improvement over the trained MoE itself. We frame this as a **training-scaffold**: the 3-decoder
MoE is necessary at training (for routing-policy gradient signal) but the middle decoder is
discardable at inference. We characterize the recipe (asymmetric warm-start, uniquely effective
when *large* is cold-init), document 8+ failure modes (Switch-aux × 4, capacity × 4, symmetric
sweep × 2), and benchmark honestly against classical LMMSE — which wins on most profiles, scoping
the contribution to "vs dense neural baseline" rather than "vs classical signal processing."

---

## 1. Problem Statement

5G neural receivers apply uniform compute per OFDM slot regardless of channel quality. Most slots
are easy (high SNR, LOS) and decodable by a small model; only a fraction need the full receiver.
Static dense receivers therefore waste energy.

**Goal:** a receiver that adapts compute per sample while keeping BLER close to a static dense
baseline. Primary metric: **BLER vs Average FLOPs** Pareto frontier on a held-out test set.
Routing must be channel-aware (no oracle SNR access).

---

## 2. Architecture

Three components:

- **Shared stem** — 2-layer MLP (hidden [64, 64], state_dim=56). Always paid (285M FLOPs).
  Produces channel features for the router AND the selected expert.
- **Channel-aware router** — small MLP (hidden 64) over **pooled stem features** (mean+max
  over freq/time), produces a categorical over experts. Gumbel-Softmax during training,
  hard top-1 at inference.
- **Heterogeneous experts** — three CNN heads with different capacities:

| Expert | block_dim | blocks | Params | Total FLOPs | % of large |
|---|---:|---:|---:|---:|---:|
| nano  |  8 | 4 |  90k |  320M |  20% |
| small | 32 | 8 | 168k |  695M |  43% |
| large | 64 | 8 | 450k | 1604M | 100% |

Total MoE: 583k params. Loss: `L = L_BCE + γ·L_channel + α·E[FLOPs] + β·L_balance`
with γ=0.05, α swept (final 2e-3), β=0.1.

---

## 3. The Journey — Three Phases of Failure, Then a Recipe

### Phase 1 — Joint training from scratch
All experts + router random init. FLOPs penalty α=1e-3.
**Result:** BLER 0.926 / 48% FLOPs. **Mode:** large is too expensive too early — router abandons
it before it can learn. Cheap, but BLER bleeds.

### Phase 2 — Full warm-start
All experts pre-trained. **Result:** BLER 0.879 / **100% FLOPs**. **Mode:** warm-large is strictly
better at step 1, router locks onto it, never explores.

### Anti-collapse mechanisms — 8 attempts, 0 wins

**Switch Transformer aux loss** (4 weights, all collapsed):

| α_aux | TDLC BLER | exp_flops | real_flops | Outcome |
|---:|---:|---:|---:|---|
| 1e-3 | 0.844 | 1.000 | 1.000 | full collapse to large |
| 1e-2 | 0.844 | 1.000 | 1.000 | full collapse to large |
| 1e-1 | 0.840 | 1.000 | 1.000 | full collapse to large |
| 1e0  | 0.859 | 0.544 | 0.996 | high-entropy soft, argmax picks large 99.6% |

**Soft capacity penalty** (4 weights, two failure modes):

| weight | TDLC BLER | real_flops | Outcome |
|---:|---:|---:|---|
| 0.1  | 0.840 | 1.000 | weak penalty → full collapse |
| 0.5  | 0.959 | 0.675 | spread routing but BLER tanks (+9pp) |
| 2.0  | 0.970 | 0.497 | very spread, BLER even worse |
| 10.0 | 0.881 | 0.600 | partial recovery, still 1pp worse than exp26 |

No middle ground exists for either mechanism where you get heterogeneous routing AND good BLER.

### Phase 3 — Asymmetric warm-start (the recipe)

Warm-start stem + nano + small. **Leave large at random init.** Large has no initial advantage; it
must *earn* router traffic by training up. At ~step 6-8k, large "wakes up." All three experts
active by step 10-12k.

**Symmetric sweep — refines the principle:**

| Setup | Avg BLER | real_flops | Outcome |
|---|---:|---:|---|
| **exp26** (cold-LARGE) | **0.902** | **0.56** | heterogeneous ✓ (44/15/40 routing on TDLC) |
| exp56 (cold-SMALL) | 0.897 | 0.82 | mostly large + some nano (small never recovers) |
| exp57 (cold-NANO) | 0.887 | 1.00 | full Phase-2 collapse to large |

**Refined finding:** asym warm-start works *specifically* when **large** is cold-init — not just
any cold expert. Mechanism: temporarily handicapping the highest-capacity expert is the only way
to break warm-large's gradient dominance and force the router to commit to smaller experts before
large becomes competent.

---

## 4. Pareto-Frontier Sprint (April 25–26)

### 4.1 Alpha sweep — find the operating point
4 jobs, asym-warm at 12k steps, seed 67.

| Run | α | Avg BLER | FLOPs % | TDLC routing l/n/s |
|---|---:|---:|---:|---|
| Dense large baseline | — | 0.901 | 100% | — |
| exp24 | 5e-4 | 0.898 | 100% | 100/0/0 (collapsed to large) |
| exp25 | 1e-3 | 0.907 | 56% | 44/12/44 (dominated) |
| **exp26** | **2e-3** | **0.902** | **56%** | **44/15/40 (knee)** |
| exp27 | 5e-3 | 0.911 | 60% | 37/0/63 (nano starved) |

**exp26 reaches dense-large BLER within 0.1pp at 56% FLOPs.** Pareto frontier is two points
(exp24, exp26); exp25/exp27 are dominated.

### 4.2 Multi-seed reproducibility — bimodal

| Seed | Avg BLER | Routing | Outcome |
|---:|---:|---|---|
| s67 (exp26) | 0.902 | 44/15/40 | ✓ |
| s42 (exp29) | 0.902 | 55/12/33 | ✓ |
| s32 (exp28) | 0.958 | 0/49/51 | ✗ large collapsed |
| s67 100k+α=2e-3 (exp40) | ~0.953 | heavy nano | ✗ |
| s42 100k+α=2e-3 (exp58) | ~0.968 | heavy nano | ✗ |
| s67 30k+α=2e-3 (exp59) | 0.926 | bad attractor | ✗ |
| **s67 100k+α=1e-3 (exp60)** | **0.901** | heterogeneous | ✓ |

**2/6 success rate at α=2e-3.** exp60 confirms: at 100k samples, scaling α inversely (1e-3 instead
of 2e-3) recovers exp26-quality BLER — **the controllable knob is the α/data ratio.** Recipe
stability at fixed α is an open problem.

### 4.3 Channel-aware router — load-bearing

exp30: identical recipe, router input is `torch.randn`.

| | TDLC BLER | Avg BLER | TDLC routing |
|---|---:|---:|---|
| exp26 (channel-aware) | 0.867 | **0.902** | 44/15/40 |
| **exp30 (random input)** | **0.965** | **0.968** | **0/11/89** (all small) |

**Random router → BLER craters 6.6pp** AND collapses to small. Channel-aware features are
load-bearing — central claim of the project, A+ confirmed.

### 4.4 Expert-count ablations

| Setup | Avg BLER | Cost vs exp26 |
|---|---:|---|
| exp26 {nano, small, large} | 0.902 | reference |
| exp31 {small, large} (drop nano) | 0.909 | +0.7pp BLER + 9pp more FLOPs |
| **exp41 {nano, large} (drop small)** | **0.955** | **+5.3pp BLER** |

Dropping small costs 7× more BLER than dropping nano. Both ablations justify the 3-expert design.

### 4.5 SNR-oracle cascade baseline

Hand-rule cascade with **true SNR** access picks the cheapest expert clearing a tolerance.

| Strategy | Avg BLER | FLOPs % |
|---|---:|---:|
| Per-profile static (best fixed) | 0.900 | 100% |
| **SNR-oracle cascade tol+0.01..0.02** | **0.900** | **49%** |
| **exp26 learned MoE** | **0.902** | **56%** |
| SNR-oracle cascade tol+0.05 | 0.909 | 35% |

With oracle SNR, hand-rule slightly dominates exp26 on FLOPs at matched BLER. exp26 sits on the
Pareto frontier *without oracle access* — motivates explicit SNR-estimate router input as future
work.

---

## 5. The Deeper Finding — What the Model Actually Learned

### 5.1 Per-expert success rate — only large decodes

For each routed sample, did the assigned expert produce a successfully decoded block?

| Expert | UMa success | TDLC success |
|---|---:|---:|
| nano | **0.00%** | **0.00%** |
| small | **0.00%** | **0.00%** |
| large | 23.21% | 29.42% |

**Nano and small literally never decode a block.** Only large delivers successful outputs.

### 5.2 Middle-expert investigation — small is a partial decoder

For each sample, we ran ALL 3 experts (forced) and recorded per-sample BER.

| Expert | BER on routed samples | BLER | Interpretation |
|---|---:|---:|---|
| nano | 0.432 | 1.0 | ≈ random — true junk on hopeless samples |
| **small** | **0.230** | 1.0 | **77% of bits correct** — partial decode but block fails |
| large | 0.025 | 0.768 | near-perfect when it works |

Counterfactual on small's routed samples: forcing large gives BER 0.231, BLER 1.0 — **large fails
on those samples too.** The router routes to small because no expert can decode them, and small
matches large's quality at half the FLOPs.

**Practical interpretation (5G context):** small's 77%-correct bits exceed the LDPC correction
threshold (~5–10% pre-LDPC BER). The block fails at LDPC and HARQ retransmission is requested.
Small's bits are not usable transmitted data — its value is **identifying samples no expert can
decode and routing to the cheapest**.

### 5.3 Mode B inference — the headline

If small never produces successful blocks, what happens if we replace its expert output with zeros
at inference (and charge zero FLOPs)?

| Mode | UMa BLER | TDLC BLER | **Avg BLER** | Avg FLOPs |
|---|---:|---:|---:|---:|
| baseline (no change) | 0.93692 | 0.86734 | **0.9021** | 55.8% |
| A_mask (force {nano, large}) | 0.93692 | 0.86722 | **0.9021** | 55.2% |
| **B_sink (zero-out small)** | 0.93692 | 0.86737 | **0.9021** | **47.3%** |

**BLER bit-identical across all 3 modes** (0.9021 to 4 decimals). **Mode B saves 8.5pp FLOPs at
zero BLER cost, with zero retraining.**

**Generalizes across the α sweep:**

| Checkpoint | baseline → B_sink BLER | baseline → B_sink FLOPs | Δ FLOPs |
|---|---|---|---:|
| exp25 (α=1e-3) | 0.9063 → 0.9066 | 56.0% → 46.7% | **−9.3pp** |
| **exp26 (α=2e-3)** | 0.9021 → 0.9021 | 55.8% → 47.3% | **−8.5pp** |
| exp27 (α=5e-3) | 0.9107 → 0.9116 | 60.0% → 41.9% | **−18.1pp** |

**Why sink-as-an-expert breaks training but works at inference:** we tried 3 architectures
training with sink as a real expert (sink+channel_only+large; sink+small+large; sink+nano+large).
**All 3 failed** — sink at 0 FLOPs is too attractive under α=2e-3, router collapses to all-sink
or training becomes unstable. The only way to use sink is **post-training substitution**.

### 5.4 The reframing — training-scaffold contribution

The compute-aware MoE is **not** "3 experts that share decoding work." It's **1 decoder (large) +
2 intelligent compute-skip mechanisms (nano: cheap fail; small: training-time scaffold)**.

> **Train: exp26 (nano + small + large)** — proven recipe, 2/6 multi-seed
>
> **Deploy: exp26 + Mode B inference** — replace small's expert with zero-output at eval
>
> **Result: avg BLER 0.9021 at avg FLOPs 47.3%** — Pareto improvement over the trained MoE itself
> at zero training cost.

This is the contribution: **a training scaffold for compute-aware inference.** Train with diverse
experts for routing-policy gradient signal; prune the non-decoding scaffold at deployment.

---

## 6. Honest Scope vs Classical LMMSE

We implemented a classical LMMSE baseline ladder (`src/baselines/lmmse.py`):

| Baseline | UMa | TDLC | Avg | Channel info |
|---|---:|---:|---:|---|
| Single-antenna | 0.992 | 0.998 | 0.995 | LS estimate, 1 antenna |
| **LS-MRC LMMSE (realistic)** | 0.939 | **0.861** | **0.900** | LS + MRC across 4 antennas |
| Genie-MRC (oracle) | 0.908 | 0.800 | 0.854 | true channel + MRC (not deployable) |

**Per-SNR head-to-head, 20 bins, TDLC waterfall:**

| TDLC SNR (dB) | LMMSE | dense_large | exp26 |
|---|---:|---:|---:|
| 14.0–15.5 | **0.7050** | 0.7069 | 0.7265 |
| 15.5–17.0 | **0.3213** | 0.3722 | 0.3697 |
| 17.0–18.5 | **0.1324** | 0.1850 | 0.1745 |
| 18.5–20.0 | **0.0502** | 0.0698 | 0.0851 |

**LMMSE wins TDLC at every waterfall bin from 14–20 dB by 2–5pp.**

**UMa** (rich multipath, NN narrowly favored at high SNR):

| UMa SNR (dB) | LMMSE | dense_large | exp26 |
|---|---:|---:|---:|
| 20.5–22.0 | 0.8100 | 0.7981 | **0.7856** |
| 22.0–23.5 | 0.7825 | **0.7551** | 0.7651 |
| 23.5–25.0 | 0.7629 | **0.7320** | 0.7423 |

**3GPP in-family OOD (TDL-A, TDL-D, CDL-A):** LMMSE 0.802 avg vs neural 0.825 — LMMSE wins.

**Per-sample crosstab on UMa:** NN net +64 wins of 32k (336 NN-only / 272 LMMSE-only / 1733 both
succeed). NN-favored subset has notably **lower spatial structure** (max eigenvalue 2.07 vs 3.22)
and **lower channel power** (0.56 vs 0.89) — NN wins specifically when MIMO geometry is weak,
where LMMSE-MRC's optimality assumptions break.

**Per-sample crosstab on TDLC:** LMMSE net +220 wins. NN-win and LMMSE-win subsets have identical
feature distributions — no clean NN-favored regime exists.

**What we DO claim:** exp26 + Mode B achieves the dense-NRX baseline BLER (0.9021) at 47% FLOPs
(vs dense_large 100%). Training-scaffold finding verified across α=1e-3 / 2e-3 / 5e-3 checkpoints.

**What we DO NOT claim:** that neural beats classical LMMSE. LMMSE wins TDLC, in-family 3GPP OOD,
and ties on UMa. The compute contribution is **vs the dense neural baseline**, not vs classical
signal processing. Neural-receiver advantages from the literature (impairment robustness, joint
estimation, pilot reduction, transferability — Honkala 2021, Wiesmayr 2023) require setups
Sionna's clean defaults don't model.

---

## 7. OOD Generalization

### 7.1 DeepMIMO ASU Campus (ray-traced 3.5 GHz)

| Model | UMa | TDLC | OOD asu_campus1 |
|---|---:|---:|---:|
| Dense large | 0.936 | 0.866 | 0.990 |
| exp26 MoE   | 0.937 | 0.867 | 0.992 |
| **LMMSE**   | 0.939 | 0.861 | (not deployable on this format) |

**All neural models fail on ray-traced.** Few-shot fine-tune (2k samples, 500 steps) did not
recover (post-FT 0.991). No catastrophic forgetting — in-dist BLER and routing preserved.

### 7.2 Why ASU fails — the synthetic prior misleads

LMMSE on O1 ray-traced (different scenario, same family): **0.976** vs dense_large 0.982 vs exp26
0.984. LMMSE *beats* both NN models on ray-traced, indicating the gap is from **learned synthetic
priors actively misleading** the receiver — not from fundamental channel difficulty.

**Substantial OOD generalization requires longer fine-tune, larger OOD slice, or
domain-randomized pretraining — out of scope for this work.**

---

## 8. Final Pareto Frontier

| Run | Avg BLER | FLOPs % | Story |
|---|---:|---:|---|
| Dense large | 0.901 | 100% | reference |
| exp24 (α=5e-4) | 0.898 | 100% | best BLER (collapsed to large) |
| exp26 (α=2e-3, baseline inference) | 0.902 | 56% | trained-MoE Pareto point |
| **exp26 + Mode B inference** | **0.9021** | **47.3%** | **the headline contribution** |
| SNR-oracle cascade tol+0.05 | 0.909 | 35% | needs oracle SNR, not deployable |

---

## 9. What Worked, What Didn't (Honest Summary)

**Worked:**
- Asym warm-start with **cold large** (uniquely effective per symmetric sweep)
- α=2e-3 at 50k samples → 0.1pp of dense at 56% FLOPs (exp26)
- α/data inverse scaling: α=1e-3 at 100k recovers exp26 quality
- Channel-aware router (random-input ablation: BLER craters 6.6pp)
- 3-expert design (drop nano: +0.7pp BLER + 9pp FLOPs; drop small: +5.3pp BLER)
- **Mode B inference**: 47% FLOPs at zero BLER cost, zero retraining, generalizes across α

**Didn't work (characterized failure modes):**
- Phase 1 / Phase 2 / β=2.0 / Switch-aux × 4 / capacity × 4 (all collapsed)
- Cold-small + cold-nano symmetric variants (only cold-large works)
- Sink-as-an-expert × 3 architectures (training-time collapse)
- Large-warmup / β-warmup stabilization (over-correct or worsen mean BLER)
- DeepMIMO few-shot fine-tune (insufficient signal at 2k samples / 500 steps)

**Open problems:**
- Recipe stability at α=2e-3 (2/6 seed reproducibility)
- OOD generalization to ray-traced channels
- Hand-rule SNR-oracle cascade slightly dominates learned router → motivates explicit SNR input

---

## 10. Comparison to Related Work

- **Wiesmayr 2024 (NRX):** static dense receiver — our dense_large baseline
- **Honkala 2021 (DeepRx):** end-to-end neural receiver, pilot-reduction story; we cite as
  motivation for why neural matters in impairment-rich regimes
- **MEAN (van Bolderik 2024):** homogeneous experts, per-SNR specialisation, no compute penalty,
  CDL-C only. We use heterogeneous experts + compute penalty + emergent routing on mixed UMA+TDLC.
  **Orthogonal contribution** — MEAN focuses on robustness, we focus on compute.
- **LOREN (van Bolderik 2026):** LoRA adapters for memory efficiency. Different axis.
- **Song et al. (2025):** channel-aware gating in wireless. We instantiate this on top of a 5G NRX
  with explicit FLOPs penalty + the training-scaffold finding.

---

## 11. Honest Scope Statement (for the report)

This work demonstrates compute-aware MoE for 5G neural receivers on **synthetic Sionna data** (UMa
+ TDL-C) and frames the contribution as a **training scaffold for compute-aware inference**:
exp26 + Mode B achieves the dense-NRX baseline BLER (0.9021) at 47% FLOPs with zero retraining.
The training scaffold finding (small expert is gradient signal during training, discardable at
inference) is verified across α = 1e-3 / 2e-3 / 5e-3 checkpoints.

**Scope:**
- vs dense neural baseline: clean Pareto win (47% FLOPs at matched BLER)
- vs classical LMMSE: scope statement, not victory — LMMSE wins TDLC and in-family 3GPP OOD;
  ties on UMa low-SNR; loses narrowly to neural at high-SNR UMa with weak MIMO geometry
- Ray-traced OOD requires fine-tune; all neural decoders fail zero-shot
- Recipe stability is bimodal (2/6 seeds at α=2e-3); α/data ratio is the controllable knob

The contribution is **to the neural-receiver literature**: *if* you've already chosen a neural
receiver (for the impairment-modeling reasons in Honkala/Wiesmayr), here is how to make it ~2×
compute-efficient via routing — and how the middle expert acts as a training scaffold that can be
pruned at deployment.

---

## Appendix A — Run Index (W&B + cluster jobs)

**Alpha sweep:** exp24 (eval `002cwsy2`), exp25 (`5jswm490`), **exp26 (`2zboo1rh`)**, exp27 (`dh4x0qmu`).

**Multi-seed:** exp28 (s32 collapsed), exp29 (s42 ✓), exp40/exp58 (100k+α=2e-3 collapsed),
**exp60 (100k+α=1e-3, ✓)**, exp59 (30k extension collapsed).

**Symmetric sweep:** exp26 (cold-LARGE ✓), exp56 (cold-small ✗), exp57 (cold-nano ✗).

**Ablations:** exp30 (random router), exp31 (drop nano), exp41 (drop small).

**Anti-collapse:** exp32–34 (large-warmup), exp35–37 (β-warmup), exp44–47 (Switch-aux),
exp48–51 (capacity), exp64/65 (sink-as-architecture).

**OOD:** dense_large + exp26 zero-shot ASU (~0.99); few-shot ft `go74dlm7`/`9t2wyyus` (no
recovery); LMMSE on O1 (0.976).

**Mode B inference (no retraining):** exp25/26/27 + neural-vs-LMMSE crosstab — see jobs
19599789 / 19599771 / 19599792.

---

## Appendix B — Operational Notes

- Single dataset (`dense-v1`) for all val/test — no mixed-protocol comparisons
- state_dim=56 non-negotiable for MoE (s32 costs 15pp waterfall BLER)
- num_workers=0 (22 GB Arrow table OOMs with worker forks on 32 GB nodes)
- One-line semantic commits, no Co-Authored-By (per project preference)
- No `gpu_mem` in qsub (model uses ~6 GB VRAM)
- Hydra `override /model: moe_X` deep-merges with `model.experts: {...}` — bake architectures
  into standalone `conf/model/<name>.yaml` files to avoid silent expert-set merge bugs
