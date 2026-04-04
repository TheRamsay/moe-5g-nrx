# MoE Roadmap

## Goal

Build a **compute-aware MoE neural receiver** that:

- keeps BLER close to the best static dense baseline,
- reduces **average FLOPs** by routing easy samples to cheaper experts,
- uses **implicit channel-quality features**, not exact oracle SNR,
- is evaluated on both in-distribution 3GPP channels and later OOD DeepMIMO.

This roadmap is based on:

- `docs/Projekt.md`
- `docs/Proposal.md`
- the current codebase state
- recent practical trends in MoE systems for efficient inference

## What Will Probably Work Best

If the goal is both a strong thesis/project and a clean engineering story, the best path is:

1. **Start from the dense baseline family you already trained**.
2. Build a **shared stem + 3 heterogeneous experts + channel-aware router**.
3. Train a **joint MoE first**.
4. Then compare it against a **staged / pretrained-expert variant**.
5. Report a **BLER vs Average FLOPs Pareto frontier**.
6. Show **expert utilization vs SNR / channel profile**.

This is strong because it is:

- technically grounded,
- experimentally clean,
- aligned with the papers you cited,
- and more interesting than a plain "MoE also works" result.

The most interesting angle for an "A" is not just MoE itself, but:

> **heterogeneous dense experts warm-started from already-trained dense baselines, routed by implicit channel-quality features, with real hard top-1 inference and measured compute-performance tradeoff**

That is much stronger than a toy homogeneous MoE.

## Why This Is Problem-Aware, Not Just "More Layers"

Your teacher's requirement is correct: a strong thesis/project should not look like:

- "we made the network bigger"
- "we added more experts"
- "we added a router because MoE is trendy"

The contribution has to be tied to the **wireless receiver problem**.

This roadmap does satisfy that requirement, but only if the implementation keeps the domain-specific pieces explicit.

### The Core Problem-Specific Ideas

The MoE should be justified by the structure of the wireless task:

1. **Channel difficulty is highly non-uniform**
   - some samples are easy and do not need a large model
   - some samples are hard because of fading / multipath / low effective SNR
   - so adaptive compute is a natural fit

2. **The input already contains physical prior information**
   - received grid
   - LS channel estimate
   - pilot structure
   - so the router can make decisions from domain-relevant evidence

3. **The cost target is physically meaningful**
   - mobile receivers care about FLOPs / energy / latency
   - so routing is not cosmetic, it is directly tied to the deployment goal

4. **The evaluation metric is domain-specific**
   - BLER is the real reliability metric
   - BLER vs Average FLOPs is much more meaningful here than generic accuracy vs params

### What Would Look Too Generic

These would be weak if presented alone:

- 3 experts with random capacities and no routing interpretation
- router trained only as an abstract classifier with no channel-quality analysis
- reporting only parameter count instead of FLOPs / utilization / SNR behavior
- claiming "adaptive compute" without showing expert usage vs channel difficulty

### What Makes It Strong

To make the project clearly problem-aware, the MoE should exploit wireless structure explicitly:

- use the **LS estimate** as part of the input and as a source of cheap routing cues
- use **pilot-aware shared features**
- normalize channel power so SNR has a consistent interpretation across profiles
- route based on **implicit channel quality**, not oracle SNR
- analyze routing against:
  - SNR
  - channel profile (`uma`, `tdlc`)
  - estimated channel reliability

In other words, the message should be:

> we are not adding more layers, we are allocating model capacity according to wireless channel difficulty using the physical structure already present in the receiver input

### Teacher-Facing Thesis Statement

If you want the project to read well to the teacher, the thesis contribution should sound like this:

> We design a domain-aware compute-adaptive neural receiver where routing decisions are conditioned on channel-quality features derived from the received resource grid and LS channel estimate, and we show that this achieves a better BLER-FLOPs tradeoff than a static dense baseline.

That is a much stronger statement than:

> We added a MoE on top of a CNN.

## High-Level Design

```text
input grid [B, 16, F, T]
    -> shared stem / feature extractor
    -> router head
    -> 3 experts (tiny / medium / heavy)
    -> gated combination during training
    -> top-1 expert only during inference
    -> logits + channel estimate
```

Recommended components:

- **Shared stem**
  - early convolutional feature extractor
  - should learn pilot-aware / channel-aware local features
- **Router**
  - lightweight head on top of shared features
  - predicts expert probabilities
- **Experts**
  - heterogeneous compute budgets
  - tiny / medium / heavy
  - same output contract as the dense model
- **Outputs**
  - bit logits
  - channel estimate

## Best Architectural Strategy For This Repo

### Reuse the Dense Capacity Sweep

This repo already has the best possible starting point for a practical MoE:

- `small`
- `mid`
- `large`

Those are not wasted baselines. They should become the **MoE expert family**.

### Recommended Expert Construction

Use the same dense model shape logic, but split it into:

- shared stem
- expert backbone
- expert readout

Since the dense family already shares the same input/output semantics and broadly similar structure, the cleanest plan is:

1. keep a common shared stem producing a fixed latent state,
2. define three expert branches with the capacities of:
   - small
   - mid
   - large
3. let the router choose among them.

### Best Initialization Strategy

This is likely the most effective and most interesting variant:

1. initialize the shared stem from the best dense baseline stem,
2. initialize:
   - tiny expert from the small dense checkpoint,
   - medium expert from the mid dense checkpoint,
   - heavy expert from the large dense checkpoint,
3. train the router and then jointly fine-tune.

Why this is strong:

- experts start competent,
- training becomes more stable,
- it reuses your dense work intelligently,
- it gives a very nice narrative in the writeup.

## Router Design

Based on your notes and the cited channel-aware routing idea, the router should not depend on explicit oracle SNR.

### Recommended Router Inputs

The router should use **shared latent features**, not hand-coded SNR only.

Good options:

- global average pooled shared stem features
- global max pooled shared stem features
- optionally a few cheap summary statistics from the input:
  - LS channel power
  - LS channel variance
  - received signal power
  - pilot-distance feature summary

Recommended first version:

- pooled shared features only

Recommended second version:

- pooled shared features + a few explicit physics-inspired statistics

This gives you a natural ablation:

- implicit router only
- implicit + explicit summary features

### Recommended Router Output

- 3 logits, one per expert
- training: Gumbel-Softmax
- inference: hard top-1 argmax routing

## Training Strategy

## Phase 0: Dense Baseline Freeze

Before MoE training, freeze the dense reference recipe:

- chosen dense capacity
- chosen optimizer recipe
- chosen dataset version
- chosen validation/test protocol

That gives you a clean target to beat.

## Phase 1: Minimal Joint MoE

This should be the **first MoE implementation**, not staged training.

Why:

- simpler to debug,
- fewer moving parts,
- proves the architecture works,
- faster route to first meaningful results.

Training behavior:

- run all experts during training,
- use router probabilities to form a weighted output,
- compute expected FLOPs from router probabilities,
- add compute penalty.

This is the easiest way to keep training differentiable.

### Recommended Loss

First working version:

```text
L = BCE(logits, targets)
  + lambda_channel * MSE(channel_estimate, channel_target)
  + alpha * expected_flops
```

Then likely add a routing regularizer:

```text
L = BCE + lambda_channel * MSE + alpha * expected_flops + beta * load_balance
```

Where:

- `expected_flops` = stem FLOPs + router_probs dot expert_FLOPs
- `load_balance` discourages collapse to only one expert too early

### Recommended Training Details

- optimizer: start from the final dense recipe
- router temperature: start around `1.0`
- anneal gradually toward `0.3-0.5`
- do not hard-route in training at first

## Phase 2: Warm-Started Experts / Two-Stage Variant

This is the variant most likely to improve quality and impress the teacher.

### Stage 2A: Expert Warm Start

- load tiny/medium/heavy expert weights from dense checkpoints
- initialize shared stem from the strongest dense stem

### Stage 2B: Router-First Adaptation

Try either:

1. freeze experts briefly, train router + shared stem,
2. then unfreeze and jointly fine-tune,

or:

1. freeze experts only for a short warmup,
2. then train everything jointly.

This directly matches your notes and proposal.

## Phase 3: Hard Inference

At inference:

- run router once,
- select top-1 expert,
- execute only that expert,
- report actual realized compute.

This part is essential. Without hard top-1 execution, the MoE compute story is weak.

## What To Implement In Code

## 1. `src/models/moe.py`

Implement at least these components:

- `SharedStem`
- `RouterHead`
- `ExpertBranch`
- `MoENRX`

Suggested forward contract:

```python
{
    "logits": ...,
    "channel_estimate": ...,
    "router_logits": ...,
    "router_probs": ...,
    "selected_expert": ...,
    "expected_flops": ...,
    "expert_outputs": ...,
}
```

During training:

- produce weighted combined outputs

During inference:

- execute only one expert

## 2. `conf/model/moe.yaml`

Expand this into a real MoE config:

- shared stem config
- router MLP config
- expert backbone/readout configs
- temperature schedule config
- routing regularization config
- FLOPs config per expert

## 3. `src/training/trainer.py`

Add support for:

- building the MoE model
- MoE loss composition
- logging router metrics
- logging compute metrics

Useful new logged metrics:

- `train/router_entropy`
- `train/expert_usage/tiny`
- `train/expert_usage/medium`
- `train/expert_usage/heavy`
- `train/expected_flops`

And later validation counterparts.

## 4. `scripts/evaluate.py`

Extend eval to report:

- per-profile BER / BLER as today
- average realized FLOPs
- expert usage histogram
- expert usage vs SNR bins

## 5. New utility module

Add something like:

- `src/utils/compute.py`

to define:

- expert FLOPs
- stem FLOPs
- expected FLOPs
- realized hard-routing FLOPs

Even a carefully documented approximate FLOP model is acceptable if it is consistent across methods.

## Experiments To Run

## Must-Have Experiments

These are the minimum experiments that produce a strong project.

### A. Dense Reference

- final tuned dense baseline

### B. Joint MoE, No Compute Penalty

Purpose:

- check whether MoE can match dense quality before forcing compute savings

### C. Joint MoE, Compute-Aware

Sweep `alpha` in:

- `0`
- small positive
- medium positive

For example:

- `0`
- `1e-5`
- `3e-5`
- `1e-4`

Goal:

- generate BLER vs Average FLOPs tradeoff

### D. Joint vs Staged Training

Compare:

- joint MoE from scratch / warm start
- expert warm start + router-first training + joint fine-tune

This is directly from your proposal and is a good scientific ablation.

### E. Expert Usage Analysis

Plot:

- expert utilization by profile
- expert utilization by SNR bin

The most convincing pattern would be:

- easy / high-SNR samples -> tiny expert more often
- hard / low-SNR or harsh channel samples -> heavy expert more often

That would strongly support your thesis story.

## High-Value Optional Experiments

### F. Router Feature Ablation

Compare:

- router from pooled latent features only
- router from latent features + cheap explicit channel statistics

This is a nice "current trends" angle because it tests implicit vs physics-guided routing.

### G. Homogeneous vs Heterogeneous Experts

Compare:

- 3 equal experts
- tiny / medium / heavy experts

This is very worth doing because your project specifically claims **heterogeneous** compute allocation.

### H. OOD DeepMIMO

Not now, but later:

- evaluate tuned dense baseline and tuned MoE on DeepMIMO
- optional small fine-tune if gap is large

## Best Thesis / "A" Story

The strongest story is probably:

1. build a strong tuned dense baseline,
2. build a heterogeneous MoE with hard top-1 inference,
3. warm-start experts from dense capacity checkpoints,
4. show BLER vs Average FLOPs tradeoff,
5. show expert utilization correlates with channel difficulty,
6. optionally show better compute robustness than dense on OOD.

This is better than a simple "MoE also works" story because it demonstrates:

- engineering depth,
- experimental discipline,
- compute-awareness,
- and interpretation of routing behavior.

## What I Think Is Most Likely To Work Best

If I had to bet on the most practical strong result:

### Best initial MoE candidate

- shared stem initialized from tuned dense baseline
- three experts with capacities matching small / mid / large
- router on pooled latent features
- joint training with compute penalty
- hard top-1 inference at eval

### Best follow-up variant

- warm-start experts from dense checkpoints
- short router warmup with experts frozen
- then joint fine-tuning

This combines:

- best chance of stable training,
- clean reuse of completed dense work,
- and a very compelling narrative.

## Risks and Pitfalls

## 1. Router collapse

Symptoms:

- all samples routed to heavy expert
- no compute savings

Mitigation:

- load-balance regularizer
- temperature schedule
- smaller `alpha` at the start

## 2. MoE beats compute but loses too much quality

Mitigation:

- sweep `alpha`
- ensure heavy expert is strong enough
- warm-start experts

## 3. Compute accounting is vague

Mitigation:

- define FLOP model explicitly and keep it fixed
- always compare dense and MoE under the same accounting rules

## 4. Too many experiments

Mitigation:

- focus on the must-have experiments first
- do not over-expand the ablation matrix before the first working MoE exists

## Recommended Execution Order

1. Freeze final dense baseline recipe
2. Implement minimal joint MoE
3. Run `alpha` sweep for BLER vs FLOPs
4. Add expert usage logging and plots
5. Add warm-started / staged training variant
6. Compare joint vs staged
7. Run final tuned MoE vs tuned dense test evaluation
8. Later: OOD DeepMIMO

## Final Recommendation

Do **not** start with the most complex staged MoE.

Start with:

- minimal working joint MoE,
- heterogeneous experts,
- expected FLOPs penalty,
- hard top-1 inference,
- expert utilization logging.

Then build the more advanced warm-started staged variant on top.

That is the shortest path to a result that is both publishable-quality for the course and realistically implementable in this repo.

## What To Emphasize In The Writeup / Presentation

If the goal is to make the project read as excellent rather than merely functional, the presentation should keep returning to the same core argument:

> channel conditions are heterogeneous, so compute should be heterogeneous too

### 1. Start From The Deployment Problem

Lead with the real systems problem:

- dense neural receivers are strong but always expensive
- mobile-side inference should not spend the same compute on easy and hard samples
- the wireless channel already gives clues about sample difficulty

That frames the MoE as a systems solution, not as architecture novelty for its own sake.

### 2. Show The Physical Prior Clearly

Explicitly highlight that your model is not working from raw pixels or arbitrary tensors.
It receives:

- the received resource grid
- the LS channel estimate
- pilot-aware spatial/temporal structure

So the router has access to information that is physically meaningful for deciding how much compute is needed.

### 3. Emphasize That Routing Is Learned, Not Oracle-Given

This is a strong contrast with weaker baselines from the literature.

Do not say:

- "the router uses SNR"

Say:

- "the router learns from shared latent channel-quality features and cheap physical priors"

That makes the approach more realistic and more interesting.

### 4. Use The Right Main Figure

The central result should not be only a BLER table.

The main figure should be something like:

- **BLER vs Average FLOPs**

Then support it with:

- expert utilization vs SNR
- expert utilization by profile (`uma`, `tdlc`)
- dense vs MoE comparison at matched reliability regions

This makes the contribution visibly compute-aware.

### 5. Tell A Routing Story, Not Just A Metric Story

The strongest interpretation section is not:

- "MoE was 0.3% better"

It is:

- easy / cleaner samples mostly activate tiny expert
- hard / degraded samples activate heavy expert
- average compute drops while reliability stays close to dense

That is what makes the method believable and domain-aware.

### 6. Reuse The Dense Work As A Design Choice

Present the dense baseline work as a deliberate foundation for MoE:

- the dense capacity sweep was not just preliminary tuning
- it identified meaningful compute tiers
- those tiers become the expert family
- this creates a principled heterogeneous MoE instead of an arbitrary one

That is a very strong story for a course project.

### 7. Include At Least One Negative / Honest Result

Teachers usually trust projects more when they show one honest tradeoff or failure mode.

Examples:

- too large compute penalty collapses routing to tiny expert and hurts BLER
- no load-balancing regularizer causes heavy-expert collapse
- homogeneous experts are less interpretable than heterogeneous ones

That shows scientific maturity.

### 8. Suggested Presentation Structure

If you later do slides or a written report, this is a good order:

1. Problem: dense NRX is accurate but always expensive
2. Insight: channel difficulty varies, so compute should vary
3. Inputs/priors: received grid + LS estimate + pilot structure
4. Method: shared stem + learned router + heterogeneous experts
5. Training: soft routing, inference: hard top-1 routing
6. Baselines: tuned dense reference and expert capacities
7. Main result: BLER vs FLOPs tradeoff
8. Interpretation: expert usage vs SNR/profile
9. Limitations / next steps: staged training, OOD, DeepMIMO

### 9. The "A" Version Of The Claim

The version you want the teacher to remember is:

> We designed a problem-aware compute-adaptive 5G neural receiver that uses channel-informed routing to spend more computation only where the wireless channel is difficult, and we validated this through BLER-FLOPs tradeoff analysis and expert-usage behavior.

If the experiments support that sentence, the project reads as coherent, domain-aware, and ambitious in the right way.
