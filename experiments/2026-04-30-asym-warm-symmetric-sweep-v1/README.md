# Asym Warm-Start Symmetric Sweep (v1)

## Hypothesis (from routing-trajectory analysis)

The success of asym-warm-start (exp26: warm nano+small, cold large) led to
the hypothesis:

> **"In heterogeneous-expert MoE, the expert with the largest initial quality
> lead becomes the routing attractor. Asymmetric initialization shifts which
> expert wins."**

This sweep tests the principle by varying which expert is cold-initialised.

## The 3 configurations

| Run | Setup | Quality at step 0 (low BLER = good) | Predicted outcome |
|---|---|---|---|
| **exp26** (existing) | warm-nano + warm-small + **cold-large** | nano 0.97, small 0.91, **large random ≈ 0.99** | Cold-large grows in by step ~10k, all 3 active |
| **exp56** (NEW) | warm-nano + **cold-small** + warm-large | nano 0.97, **small random ≈ 0.99**, large 0.87 | Router may collapse to large (no negative gap), small may not survive |
| **exp57** (NEW) | **cold-nano** + warm-small + warm-large | **nano random ≈ 0.99**, small 0.91, large 0.87 | Most pessimistic: cold expert is the smallest, no capacity advantage to grow into |

## Why these tests matter

If exp56 and exp57 BOTH still produce heterogeneous routing → "cold-expert
grows in" is a robust mechanism, asym-warm generalizes.

If exp56 OR exp57 collapses to large → the principle is more nuanced. Likely
explanation: large NEEDS to be the cold expert because its capacity advantage
is what eventually pulls traffic back. Smaller cold experts lack this draw.

If BOTH collapse → confirms exp26's specific recipe (cold-LARGE) is privileged
because of large's capacity dominance, not the general principle.

This is a falsifiable test of the trajectory-analysis hypothesis. Either
outcome is publishable — and shapes the writeup of the asym-warm finding.

## Cluster

- Resources: `select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb`
- Walltime: 3h each (12k steps ~ 40 min compute + buffer)

## Status

Both jobs (19586392, 19586393) finished 2026-04-30 evening.

## Results

| Exp | Setup | UMa BLER | TDLC BLER | Avg BLER | real_flops | Outcome |
|---|---|---:|---:|---:|---:|---|
| **exp26** (cold-LARGE) | warm nano+small, cold large | 0.937 | 0.867 | 0.902 | 0.56 | heterogeneous ✓ (44/15/40) |
| **exp56** (cold-SMALL) | warm nano+large, cold small | 0.939 | 0.856 | 0.897 | **0.82** | mostly large + some nano (small never recovers) |
| **exp57** (cold-NANO) | warm small+large, cold nano | 0.931 | 0.843 | 0.887 | **1.00** | **FULL Phase-2 collapse to large** |

## Refined principle

Original hypothesis: *"warm experts win the routing lottery"* (too symmetric).

**Refined finding:** asym-warm-start is uniquely effective when LARGE is
cold-init. Other "asym" configurations don't produce heterogeneous routing.

Mechanism:
- **exp26 (cold-large):** warm nano (0.97) and small (0.91) are temporarily
  better than random large (0.99). Router uses them. Large hidden until it
  catches up (~step 8-10k). Final state: heterogeneous.
- **exp56 (cold-small):** warm large (0.87) is best from step 0. Router
  prefers large. Cold small never recovers (real_flops=0.82, mostly large).
- **exp57 (cold-nano):** same dynamic — warm large dominates from step 1,
  router commits to it, FULL collapse to 100% large.

## Implications for the writeup

This is a **stronger publishable claim** than just "asym-warm works":

> *"The asymmetric warm-start recipe is uniquely privileged when LARGE
> specifically is cold-initialized. The mechanism is that temporarily
> handicapping the highest-capacity expert forces the router to commit to
> smaller experts before the most-capable one becomes available. Inverting
> which expert is cold (exp56, exp57) does not produce heterogeneous routing
> — both lead to large-dominated configurations. This explains why the
> recipe works AND why it can't be trivially generalized."*

This finding ties together:
- The trajectory analysis (Phase-2 collapses immediately, asym-warm delays commitment)
- The bimodal seed result (occasional collapse even with cold-large is a
  related instability)
- The 100k collapse (where stronger α tipped the balance even under cold-large)
