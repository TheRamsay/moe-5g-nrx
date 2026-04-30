# Post-Consultation Notes — 2026-04-30

**Outcome:** got roasted. Key feedback below + action plan for the next 13 days
before the final deadline.

---

## What the teacher said

### 1. Explain all parts of the NN in detail
> *"He really wanted us to explain all parts of NN in detail and I couldn't."*

The teacher expects we can walk through every component of the architecture,
not just hand-wave at "it's a MoE." This means:

- Input shape, what each channel represents, why
- Every layer's job, dimensions in/out, activation
- Why each design choice (kernel size, hidden dim, pooling, etc.)
- What each loss term penalises and why we need it
- How training differs from inference (Gumbel-Softmax → hard top-1)
- How the FLOPs penalty actually works mathematically

**This is non-negotiable for the final defense — be able to draw the whole
architecture on a whiteboard from memory.**

### 2. Standard baseline comparison
> *"He wants some standard baseline to compare it against (so non-neural
> approaches or get some NN approach and benchmark it)."*

Our current comparison is **only** against our locally-retrained dense
baseline. The teacher wants one or both of:

- **Non-neural classical baseline:** e.g. LMMSE channel estimation +
  classical symbol detection. This is the legacy 5G receiver pipeline that
  any neural receiver paper should compare against. Sionna can simulate this.
- **Other neural baseline:** e.g. MEAN (van Bolderik 2024) reimplementation
  with homogeneous experts, or full Wiesmayr replication.

This is the **biggest concrete gap** — we explicitly cut MEAN as
"too time-expensive" in the roadmap, and we have no classical baseline.

### 3. Clean methodology, no random assumptions
> *"Clean methodology, whys, no random assumptions."*

Every design decision must be **justified with either**:
- A reference (paper / theory / standard)
- Empirical evidence (an ablation we ran)
- A clearly-stated assumption (when neither of the above is possible — and
  we mark it as an assumption, not a fact)

No more "I think this is because..." or "we just chose this."

Specific things this likely refers to:
- Why these expert sizes? (We need: matched pre-trained baselines, 2-expert
  ablation evidence)
- Why Gumbel-Softmax? (Standard for differentiable discrete routing)
- Why α=2e-3 specifically? (Empirical from sweep — show the sweep)
- Why MSE vs BCE? (Output type — regression vs classification)
- Why mean+max pooling for router? (Standard global summary; could ablate)
- Why state_dim=56? (Standard from Wiesmayr; we have a separate s32 vs s56
  experiment showing 56 is needed)

### 4. He will ask all sorts of questions
> *"He will ask all sorts of questions."*

Be ready for anything. Specifically prep for:
- "Why didn't you do X?" → know what we cut and why
- "What if you change Y?" → know our parameter sensitivities
- "How does this compare to Z paper?" → know our comparison gaps
- "Show me the math for [some loss term]" → have it ready
- "What's your single biggest weakness?" → name it before he does
- "What would you do with another month?" → concrete prioritised list

---

## Action plan — 13 days to deadline

### Priority 1 (must do): NN explanation mastery
**Time:** 2-3 hours self-study + practice

- Re-read `src/models/moe.py` line by line
- Draw the architecture diagram from memory 3+ times
- Practice the 30-second elevator + 5-minute deep-dive narration
- Have a printed cheatsheet of dimensions at each layer

### Priority 2 (must do): Add a standard baseline
**Time:** 2-3 days

Two options, pick one (or both if time allows):

**Option A — LMMSE classical baseline (cleaner, faster)**
- Implement classical LMMSE channel estimation + symbol detection
  pipeline using Sionna's built-in receivers
- Evaluate on the same UMa + TDLC test sets at the same SNR bins
- Add to the Pareto figure as a non-neural reference point
- Frames our work as "neural receivers vs classical, plus our compute
  optimisation"

**Option B — MEAN reimplementation (heavier, more direct)**
- 3 same-size experts (homogeneous), per-SNR specialisation gating
- Train on same data, eval on same metrics
- Direct comparison: "their MoE design vs ours, same data"
- This was our cut "Optional A+" item; uncutting it now matches the
  teacher's request

**Recommendation:** Do **A first** (LMMSE) — it's the universal baseline
every NRX paper has. If time permits, also B.

### Priority 3 (must do): Methodology audit
**Time:** 1-2 days

Go through every design decision in the report and confirm each has either
a reference, empirical evidence, or an explicit assumption marker. Specific
audit checklist:

- [ ] Expert sizes — justify (matched dense baselines + 2-expert ablation)
- [ ] state_dim=56 — justify (Wiesmayr standard; have ablation?)
- [ ] α=2e-3 — justify with the sweep figure
- [ ] β=0.1 — justify (avoided collapse, didn't hurt BLER; show alternatives)
- [ ] γ=0.05 — justify (Wiesmayr standard)
- [ ] Asym warm-start — justify (5 anti-collapse mechanisms documented)
- [ ] Gumbel-Softmax (training) → hard top-1 (inference) — justify (standard)
- [ ] Mean+max pooling — justify (standard global summary; or ablate it)
- [ ] 3 experts vs N — justify (2-expert ablation)
- [ ] FLOPs penalty form `E[FLOPs ratio]` — justify (smooth differentiable)
- [ ] SNR bin selection (7 bins) — justify (eval cost vs resolution; flag
      as future work for finer sampling)

### Priority 4 (high): The three concrete future-work experiments
We already have these as the future-work plan:

1. **Proper inference latency benchmark with efficient dispatch**
2. **High-resolution per-SNR evaluation** (denser SNR sampling 10-20 dB)
3. **Router interpretability** (PCA of stem features by selected expert + SNR)

Pick at least one to actually do before the final deadline, not just
list as future work. Each is doable in 1-2 days. The PCA interpretability
is probably the highest-value-per-effort.

### Priority 5: Q&A drilling
- Practice answering the likely questions out loud
- Pre-mortem: write down every question you're afraid of, prepare answers
- Have someone (teammate?) play devil's advocate

---

## Specific weak points to fix

### "Why these expert sizes?"
**Old answer:** "We picked them."
**New answer:** *"We matched the sizes of pre-trained dense baselines
(nano/small/large) so we could warm-start the experts from existing
checkpoints. The 5× compute spread (20% / 43% / 100%) gives meaningful
heterogeneity — narrower spreads (43% / 69% / 100% in our original
small/mid/large config) failed because the router never preferred mid over
large. The 2-expert ablation confirms nano isn't redundant — dropping it
costs 0.7 pp BLER and 9 pp FLOPs."*

### "How do you know your model beats classical baselines?"
**Currently:** we don't.
**Fix:** Implement LMMSE baseline (Priority 2 above).

### "How does this compare to MEAN?"
**Old answer:** "Orthogonal contribution."
**New answer:** Same, but acknowledge the gap: *"We argue this is
architecturally orthogonal — MEAN's homogeneous experts can't reduce
compute, only specialise. We don't have a direct numerical comparison;
that reimplementation was on our roadmap and is the highest-priority
follow-up if time permits."*

### "Why is your average BLER ~0.9? That seems terrible."
**New answer:** *"It's averaged across the full SNR sweep, including
low-SNR regions where every model fails (BLER=1.0). The relevant comparison
is to dense_large under identical SNR distribution — 0.902 vs 0.901, within
noise. Per-SNR breakdowns show the comparison works correctly bin-by-bin."*

### "Why are you only testing on synthetic Sionna data?"
**New answer:** *"Standard scope for this work. We did test on DeepMIMO
ray-traced data as OOD — all models including dense fail (~0.99 BLER), and
brief few-shot fine-tune was insufficient. We report this transparently as
a scope limitation; bridging synthetic-to-ray-traced is a separate research
problem requiring more data and training time."*

### "What's your single biggest weakness?"
**Honest pick:** *"Lack of a standard non-neural or alternative-neural
baseline. We compare against our retrained dense baseline, but a proper
characterisation should include LMMSE classical receiver and ideally MEAN
reimplementation. We are addressing this in the final 13 days."*

(Saying this *before* the teacher says it = strength, not weakness.)

---

## Things NOT to do tomorrow

- Don't oversell. Honest scope statements beat "this works perfectly!"
- Don't claim "1.93× wall-clock speedup" — we know that doesn't hold on
  real data with our naive dispatch; FLOPs is the right metric.
- Don't claim MEAN comparison without doing it.
- Don't promise things you can't deliver in 13 days.

---

## What a successful final defense looks like

- Confidently walk through the full architecture, dimensions and all
- Show Pareto plot with at least one non-neural baseline (LMMSE)
- Acknowledge limitations *before* being asked
- Have done at least one of the three future-work experiments (probably
  router interpretability)
- Answer "why X?" for every design decision with reference/evidence/
  marked-assumption
- Be the one bringing up MEAN comparison gap, not the teacher

---

## Open questions for the team

- Who implements the LMMSE baseline? (Sionna has built-ins — couple days)
- Do we attempt MEAN reimplementation, or just LMMSE?
- Which interpretability experiment first — PCA, or something else?
- Final report writing schedule — start tomorrow, finish 5 days before?
