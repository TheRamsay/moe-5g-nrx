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

Submitted 2026-04-30 evening. Two jobs in flight.
