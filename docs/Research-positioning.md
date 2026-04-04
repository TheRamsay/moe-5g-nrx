# Research Positioning

## Why This Project Is Research-Relevant

This project is not just an engineering exercise or a larger neural network.
It addresses a real open systems question in learned wireless receivers:

> can we keep the reliability of a strong static dense neural receiver while reducing average inference compute by adapting model capacity to channel difficulty?

That question is research-relevant because it combines:

- **modern neural physical-layer receivers**,
- **adaptive inference / mixture-of-experts**,
- **wireless-domain physical priors**,
- **systems-aware evaluation** through BLER vs Average FLOPs.

The project becomes clearly stronger than a generic deep-learning exercise if we keep the contribution centered on the wireless problem itself:

- channel conditions are heterogeneous,
- compute needs are heterogeneous,
- the receiver already has access to physical structure (`LS` estimate, pilot layout, received grid),
- and the deployment objective is not only accuracy, but also compute efficiency.

## Core Research Claim

The strongest and most defensible project claim is:

> We design a compute-aware, channel-informed MoE neural receiver that uses physically meaningful channel-quality cues to route samples to heterogeneous experts, and we show a better BLER-FLOPs tradeoff than a static dense baseline.

This is stronger than:

- "we added a MoE"
- "we added more layers"
- "we trained a different architecture"

because it explicitly ties the architecture to the wireless setting and the compute-efficiency goal.

## Where It Sits Relative To The Papers

Based on `docs/Projekt.md` and `docs/Proposal.md`, the most relevant comparisons are:

### 1. Wiesmayr et al. (2024) / dense 5G NRX baseline

Role in your project:

- defines the modern dense neural receiver reference point,
- motivates the architecture family you already trained,
- provides the baseline that MoE must compete against.

What you should take from it:

- strong static dense receiver as the quality target,
- same task framing (resource-grid-to-bit-logit prediction),
- dense reference as the upper-quality / upper-compute baseline.

What your project adds beyond it:

- adaptive compute,
- heterogeneous experts,
- routing,
- compute-aware evaluation.

### 2. MEAN (van Bolderik et al., 2024)

Role in your project:

- proof that MoE ideas are meaningful in 5G receivers,
- conceptual precedent for routing in this domain.

What your notes already identify as a limitation:

- homogeneous experts,
- reliance on exact SNR / unrealistic routing information.

What your project should do differently:

- heterogeneous experts,
- no oracle SNR requirement,
- router driven by learned channel-quality features from actual model input.

This is one of your clearest research differentiators.

### 3. Song et al. (2025) / channel-aware gating

Role in your project:

- supports the idea that routing should be channel-informed,
- justifies a router conditioned on shared latent channel-quality features.

What to borrow conceptually:

- gating should reflect channel state / channel difficulty,
- routing should not be arbitrary.

What your version should emphasize:

- implicit channel-aware routing from the resource grid and LS estimate,
- practical compute-aware inference.

### 4. LOREN (van Bolderik et al., 2026)

Role in your project:

- points toward memory-efficient adaptation and modular neural receiver design,
- supports the idea that receiver specialization can be useful.

What to borrow conceptually:

- efficient adaptation matters,
- modular structure matters.

What not to overpromise yet:

- do not try to match all of LOREN's complexity in the first MoE version,
- keep your first result focused on hard top-1 routing and compute-aware tradeoff.

## The Right Comparison Strategy

The most important research practice is:

> do not compare your MoE only to numbers from papers with different simulators, datasets, or evaluation settings.

Instead, use **two layers of comparison**.

### A. Primary comparison: reproduced local baselines

This should be your main evidence.

Use:

- your own tuned dense baseline,
- your own dense capacity sweep,
- same Sionna pipeline,
- same cached validation/test sets,
- same metric definitions,
- same FLOP accounting rules.

This is the fairest and strongest comparison.

### B. Secondary comparison: literature positioning

Use the papers to explain:

- why the problem matters,
- what design choices are inspired by prior work,
- what your contribution changes relative to them.

Do **not** overclaim direct numerical superiority unless the setup is truly comparable.

## Good Research Practices For This Project

These are the practices that make the project look rigorous and "research-grade":

### 1. Freeze dataset versions

For each study phase, use one fixed cached validation/test dataset version and reuse it across runs.

That is already how you are moving now with:

- `dense-v1/val`
- `dense-v1/test`

### 2. Separate train / validation / test cleanly

- train: on-the-fly Sionna mixed generation
- validation: cached `uma` + `tdlc`
- test: cached `uma` + `tdlc`

Do not tune on the test set.

### 3. Use stage-wise experimentation

This is already the right direction:

1. dense baseline
2. dense capacity sweep
3. hyperparameter sweep on winner
4. then MoE

That isolates variables and makes conclusions believable.

### 4. Compare with the same budget first

For architecture comparisons:

- same training budget,
- same validation frequency,
- same optimizer recipe,
- same data version.

Only after that should you do final tuning.

### 5. Report both quality and compute

For the MoE phase, do not stop at BER/BLER only.

You should report:

- BLER
- BER
- Average FLOPs
- BLER vs FLOPs
- expert utilization

Without the compute side, the MoE story is incomplete.

### 6. Use hard-routing at inference

This is essential.

If all experts are still effectively active at inference, the claimed compute savings are weak.

### 7. Keep a clear ablation structure

At minimum, the MoE study should separate:

- dense baseline,
- MoE without compute penalty,
- MoE with compute penalty,
- joint training,
- staged / warm-started training.

### 8. Track lineages and study metadata

You already have a strong start here:

- W&B artifacts,
- train/eval separation,
- experiment study folders,
- registry metadata.

This is exactly the kind of infrastructure that supports a good research narrative.

### 9. Use multiple seeds for the final chosen model

Not necessary for every exploratory run, but for the final dense baseline and final MoE result, try to run at least a small multi-seed confirmation.

Even `3` seeds is much better than reporting only one lucky run.

### 10. Include one honest failure mode

A strong research report usually includes one transparent negative result or tradeoff, for example:

- too strong FLOPs penalty collapses to tiny expert,
- homogeneous experts are less interpretable,
- random router features underperform channel-aware ones.

That shows scientific maturity.

## Recommended Comparison Table

For the final report or thesis, one of the most useful tables would be:

| Model | Params | Avg FLOPs | UMa BER | TDL-C BER | UMa BLER | TDL-C BLER | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Dense baseline | ... | ... | ... | ... | ... | ... | tuned reference |
| MoE alpha=0 | ... | ... | ... | ... | ... | ... | quality-first MoE |
| MoE alpha=... | ... | ... | ... | ... | ... | ... | compute-aware |
| MoE staged | ... | ... | ... | ... | ... | ... | warm-start variant |

That table is much more convincing than only a list of runs.

## Recommended Figures

The strongest set of figures would be:

### Figure 1: BLER vs Average FLOPs

This should be the headline result.

### Figure 2: Expert utilization vs SNR

Shows whether the router actually learned a meaningful policy.

### Figure 3: Expert utilization by profile (`uma`, `tdlc`)

Supports the domain-aware routing story.

### Figure 4: SNR-binned BER/BLER curves

Lets you see where compute savings help or hurt.

## Strongest Final Story For The Teacher

The cleanest "A-level" story is:

1. you built and tuned a trustworthy dense baseline,
2. you turned the dense capacity hierarchy into a heterogeneous expert family,
3. you built a channel-aware router using physical prior information already present in the receiver input,
4. you showed a meaningful BLER-FLOPs tradeoff,
5. you interpreted routing behavior rather than only reporting numbers.

That is not just deep learning. It is a proper domain-aware ML systems project.

## Short Practical Conclusion

Yes, the project is clearly research-relevant **if** you keep the contribution centered on:

- adaptive compute,
- wireless-domain priors,
- realistic routing,
- and fair dense-vs-MoE comparison under the same evaluation pipeline.

The strongest comparison is not directly against published absolute numbers.
The strongest comparison is:

- your reproduced dense baseline,
- your tuned dense winner,
- your MoE variants,
- all measured under the same Sionna + cached-eval setup.
