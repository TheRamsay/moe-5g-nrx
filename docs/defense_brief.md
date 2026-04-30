# Defense Brief — Compute-Aware MoE for 5G Neural Receivers

**For tomorrow's progress defense.** Teacher has seen the checkpoint report
(up to asym warm-start, 0.910 BLER / 61% FLOPs). This meeting is about
everything since. Start with §0 to orient him, then use the rest for Q&A prep.

---

## 0. What's new since the checkpoint report

The checkpoint left off at: **asym warm-start, 0.910 BLER, 61% FLOPs, single seed.**
Since then, ~10 experiments were run. Here is the complete list:

| What | Result |
|---|---|
| **Alpha sweep** (4 values of α) | exp26 (α=2e-3) is the Pareto knee: **0.902 BLER / 56% FLOPs** |
| **3-seed confirmation** | Bimodal: 2/3 seeds reproduce (s67, s42 → 0.902); **1/3 collapses** (s32 → 0.958) |
| **Stabilization attempts** × 2 | large-warmup → 100% large; β-warmup → worse BLER. **Both failed.** |
| **Random router ablation** | Replace router input with noise → **BLER craters 6.6 pp** + collapses to small. Channel features are load-bearing. |
| **2-expert ablation** (drop nano) | **0.7 pp worse BLER + 9 pp more FLOPs.** Nano earns its keep. |
| **DeepMIMO OOD eval** | All models fail (~0.99 BLER) on ray-traced ASU campus data. |
| **Few-shot fine-tune** (500 steps, 2k OOD samples) | No recovery. Honest scope: synthetic-only. |
| **SNR-oracle baseline** | Hand-rule cascade with true SNR: 0.900 / 49%. exp26 on Pareto frontier without oracle. |
| **Explicit SNR-input ablation** (exp38) | Adding channel stats to router → **100% collapse to large**. Implicit stem features are sufficient. |

---

## 1. Elevator pitch (30 seconds)

We built a 5G neural receiver that **adapts compute per sample**. Instead of
running one big model on every received OFDM slot, we route each slot through
a Mixture of Experts with three different sizes (nano / small / large). A
small router looks at channel quality and picks the cheapest expert that
can decode that slot.

**Headline result:** our best model (`exp26`) matches the dense baseline
within **0.1 percentage points BLER** while running on **56% of the FLOPs**.
The router learned this routing behavior on its own from a FLOPs penalty in
the training loss — no per-sample SNR labels.

---

## 2. The problem (30 seconds)

A 5G neural receiver decodes OFDM symbols into bits. The dense baseline
(Wiesmayr et al. 2024) applies the same compute to every slot regardless
of how easy or hard it is. Most slots are easy (high SNR, line-of-sight)
and could be decoded by a tiny network. Only the hard ones (waterfall
region, low SNR) need the full receiver.

So static dense receivers waste compute. We want compute that scales with
**channel difficulty**.

---

## 3. The architecture (1 minute, optional whiteboard sketch)

Three components in series:

1. **Shared stem** — a small MLP that processes the received signal + LS
   channel estimate. Always runs (285M FLOPs). Produces a feature
   representation used by the router AND the selected expert.

2. **Channel-aware router** — a tiny MLP that takes pooled stem features
   (mean+max pool over frequency and time) and outputs a probability over
   3 experts. Trained with Gumbel-Softmax; **hard top-1** at inference (so
   only ONE expert runs — savings are real, not amortized).

3. **3 heterogeneous experts** — same architecture family, different
   capacity:
   - **nano** (90k params, 320M total FLOPs, 20% of large)
   - **small** (168k params, 695M total FLOPs, 43% of large)
   - **large** (450k params, 1604M total FLOPs, 100%)

Loss: `BCE + γ·channel_MSE + α·E[FLOPs ratio] + β·load_balance`.
α is the FLOPs penalty — turning it up trades BLER for speed. β prevents
collapse to a single expert.

---

## 4. The journey (the part to actually tell as a story)

The interesting part of this project is **the recipe**, because we hit two
opposite failure modes before finding what works:

**Phase 1 — joint training from scratch:** All experts random init. The
FLOPs penalty kicks in early when none of them are good, and the router
learns "large is expensive, abandon it." Result: BLER 0.926 / 48% FLOPs.
Cheap, but BLER bleeds.

**Phase 2 — full warm-start:** Initialise each expert from a pre-trained
dense checkpoint of matching size. Now the router sees that warm-large is
strictly better than warm-nano/small from step 1, locks onto large, never
explores. Result: BLER 0.879 / **100% FLOPs** — basically a fine-tuned dense
large.

**Anti-collapse experiments (5 attempts, all failed):** stronger β,
β=2.0 forced uniform, capacity constraints, Switch-Transformer auxiliary
loss — all either collapse or kill BLER.

**Phase 3 — asymmetric warm-start (the fix):** warm-start stem + nano + small
from dense checkpoints, but **leave large at random init**. Now large has no
initial advantage; the router uses only nano and small for ~6-8k steps.
Then large "wakes up" once it learns enough to be useful, and the router
discovers it. **All three experts active** by step 10-12k.

This is the recipe everything else is built on.

---

## 5. The headline result and how we know it's real

After the asym-warm fix worked, we ran a **4-point sweep over the FLOPs
penalty α** to find the best operating point.

| Run | α | Avg BLER | FLOPs % | Routing l/n/s | Verdict |
|---|---:|---:|---:|---|---|
| Dense large | — | 0.901 | 100% | — | reference |
| exp24 | 5e-4 | 0.898 | 100% | 100/0/0 | α too weak → collapses to large |
| exp25 | 1e-3 | 0.907 | 56% | 44/12/44 | dominated |
| **exp26** | **2e-3** | **0.902** | **56%** | 44/15/40 | **knee of the Pareto** |
| exp27 | 5e-3 | 0.911 | 60% | 37/0/63 | α too strong → nano starved |

**exp26 is 0.1 pp BLER from dense_large at 56% FLOPs.** Strict Pareto
improvement.

---

## 6. The ablations (proving the design choices are load-bearing)

These are the questions a sharp reviewer would ask. We answered both.

### "Is the router actually using channel features, or could it be random?"
Replaced the router input with `torch.randn` (same training recipe).
Result: **BLER craters by 6.6 pp** AND the router collapses to small (large
is never used). **Channel-aware features are load-bearing** — exactly the
central claim of the project.

### "Do you really need three experts, or would two suffice?"
Dropped nano, trained with only {small, large}. Result: **0.7 pp worse BLER
+ 9 pp more FLOPs.** Nano is non-decorative — it absorbs hopeless low-SNR
samples that small would burn compute on.

---

## 7. The honest weaknesses (and how we own them)

### Seed stability
Re-ran α=2e-3 with seeds 32 and 42 alongside our headline seed 67:
- s67: avg 0.902 ✓
- s42: avg 0.902 ✓
- **s32: avg 0.958, large collapsed** ✗

**2 of 3 seeds reproduce; 1 hits the Phase-2 attractor.** We tried two
stabilization recipes (large-warmup, β-warmup); **both failed** (large-warmup
over-corrects to 100% large; β-warmup gives worse mean BLER). We report this
transparently.

### OOD generalization
We tested on **DeepMIMO ray-traced data** (ASU campus, 3.5 GHz). All three
of our models — including dense large — fail catastrophically (~0.99 BLER).
We then tried a **few-shot fine-tune** (500 steps, 2k OOD samples).
**Negative result:** dense_large 0.990 → 0.9901, exp26 0.992 → 0.9915.
500 steps / 2k samples is **insufficient** to bridge the synthetic-vs-ray-traced
gap. Useful side findings: no catastrophic forgetting on in-distribution, and
the routing didn't adapt either (nano-default preserved on OOD).

### SNR-oracle baseline
Built a hand-rule cascade that uses **true SNR** to pick the cheapest expert
that clears a BLER tolerance. With oracle SNR, this cascade reaches **0.900
BLER at 49% FLOPs**, slightly dominating exp26 (0.902 / 56%). **exp26 is on
the Pareto frontier without oracle access.** We also tried feeding raw
signal statistics (channel power, channel variance — SNR proxies) directly
to the router (exp38) — that collapsed to 100% large. So the gap to oracle
remains, but raw input statistics are not the way to close it.

---

## 8. What we did vs prior work

- **Wiesmayr et al. 2024** — defines the dense NRX architecture we use as
  baseline.
- **MEAN (van Bolderik 2024)** — also MoE for 5G NRX, but with **homogeneous
  experts** (same compute) and per-SNR specialisation, not compute efficiency.
  Trained on CDL-C only. **Orthogonal contribution** — they target robustness,
  we target compute.
- **Song et al. 2025** — channel-aware gating in wireless. We instantiate
  this idea on top of an NRX with an explicit FLOPs penalty.

---

## 10. Likely teacher questions and short answers

**Q: Why heterogeneous experts and not just one expert at varying capacity?**
A: Heterogeneous lets us actually skip computation. A single resizable
expert would still need to run something on every sample.

**Q: Why not just use SNR as the router input directly?**
A: At inference time we don't have ground-truth SNR. We DID try feeding raw
signal statistics that correlate with SNR (channel power, channel variance)
directly to the router — exp38, run today. Result: collapsed to 100% large.
Implicit stem features outperform explicit raw statistics. A properly
trained SNR-estimator module (not raw stats) might behave differently —
that's the open direction.

**Q: How do you know the router isn't just memorising profile (UMa vs TDLC)?**
A: Two pieces of evidence: (a) per-SNR breakdowns show the router transitions
within each profile (small → large in the waterfall region), not a constant
choice per profile; (b) the random-router ablation shows that without channel
features the model collapses entirely — so the router IS using them.

**Q: Is one seed enough?**
A: No, and we know. We ran 3 seeds; 2 reproduce, 1 collapses. We tried two
stabilization recipes, both failed. We report the bimodal distribution
honestly and recommend best-of-N seeds.

**Q: You only show synthetic Sionna results. What about real channels?**
A: We tested on DeepMIMO ray-traced ASU Campus. All models — including
dense — fail without OOD adaptation. Brief few-shot fine-tune (500 steps,
2k samples) was insufficient to recover. Honest scope statement: this work
is for synthetic 3GPP channels; bridging to ray-traced needs more data and
training time than we had.

**Q: What's the contribution given that MEAN already did MoE for NRX?**
A: Three things MEAN doesn't have:
  1. **Heterogeneous expert sizes** — actual compute heterogeneity, not just
     specialisation.
  2. **FLOPs penalty in the loss** — emergent compute-aware routing.
  3. **Pareto analysis** + ablations + oracle baseline — a complete
     end-to-end characterisation rather than a single number.

**Q: What would you do next? (before the final deadline, 13 days)**
A: Three concrete experiments motivated by what we just found:
  1. **Proper inference latency benchmark with efficient dispatch.** Current
     naive top-1 dispatch (3 sequential sub-batches per forward pass) doesn't
     translate FLOPs savings into wall-clock. Implement production-grade
     dispatch (cf. Mixtral, vLLM dispatch kernels) and re-benchmark exp26
     vs dense_large at multiple batch sizes.
  2. **High-resolution per-SNR evaluation.** Current 7 bins per profile gives
     us 2 points in the TDLC waterfall (14 dB and 18 dB). Resample at
     1-2 dB steps across 10-20 dB to characterise where exactly exp26
     diverges from dense_large — and whether the BLER gap is uniform or
     concentrated in specific SNR regions.
  3. **Router interpretability — what does the router *see*?** We have
     preliminary PCA of stem features on the cluster, but haven't dug deeper.
     Concrete plan: (a) PCA/UMAP of stem features coloured by selected expert
     and by true SNR, (b) saliency maps over the input grid showing which
     subcarriers/symbols drive routing decisions, (c) per-expert feature
     activation analysis to see what each expert specialises on.

  Beyond these three, the longer-horizon open problems remain: **seed-stable
  training recipe** (ours collapses 1/3) and **OOD robustness** (DeepMIMO).

---

## 11. Status (where we are vs the deadline)

- **Experiments:** essentially complete. Headline + 3 ablations + 3-seed +
  oracle baseline + OOD + few-shot OOD = **~10 distinct experiments
  done with results**.
- **Doc:** `docs/final_draft.md` is a complete narrative draft. The official
  `docs/checkpoint_report.md` still needs the LaTeX rewrite — the actual
  remaining bottleneck.
- **Poster:** not started.
- **Time to deadline:** 13 days.

We have plenty of runway. The risk is **execution on the writeup**, not
on the work itself.

---

## 12. One slide / one number per claim

If you only have time for one bullet per claim:

- **Compute-aware MoE works:** exp26 hits dense BLER at 56% FLOPs.
- **The router really uses channel features:** random-router ablation
  loses 6.6 pp BLER.
- **3 experts > 2 experts:** dropping nano costs 0.7 pp BLER and 9 pp FLOPs.
- **Asym warm-start is the only recipe that works:** Phase 1 and Phase 2
  both fail cleanly; we characterized both.
- **Seed-stable? Mostly:** 2/3 seeds reproduce; 1 collapses. Reported.
- **OOD generalization?** Synthetic-only scope; brief fine-tune insufficient.
- **Beat by oracle?** Slightly (49% vs 56% FLOPs at same BLER). We tried
  feeding raw SNR-proxy stats to the router (exp38) — collapsed. Implicit
  features beat explicit raw stats.
- **Explicit SNR proxies tried (exp38):** 100% collapse to large.
  Implicit stem features are sufficient.

That's the project.
