# Bold Simplification — `{sink, nano, large}` (exp65)

## Hypothesis

Middle-expert investigation (2026-05-01) showed **nano can partial-decode borderline samples**:

| Forced expert on small's routed samples | BER | FLOPs |
|---|---:|---:|
| nano | 0.237 | 320M |
| small | 0.230 | 695M |
| large | 0.231 | 1604M |

Nano's BER is virtually identical to small's. So at INFERENCE, nano could
substitute for small at half the FLOPs. **The question is whether the router
can train to use nano in BOTH regimes** (random output on hopeless samples,
partial decode on borderline samples) without small as a separate option.

## Why it might work
- Nano IS partial-decoding borderline samples (when forced) — capability exists
- Channel-MSE auxiliary loss touches all experts, encouraging dual-regime learning
- Router sees the same channel-aware features regardless of expert count

## Why it might fail (and that would also be a useful finding)
- exp41 (drop small at training): +5.3pp BLER hit. If exp65 hits a similar
  regression, confirms the irreducibility of the 3-tier design.
- Routing dynamics may not converge to dual-regime nano usage.

## Architecture vs alternatives

| Architecture | Total params | Expert FLOPs (sink/mid/large) | Predicted |
|---|---:|---:|---|
| exp26 (nano + small + large) | 583k | 320M + 695M + 1604M | reference |
| exp64 (sink + small + large) | 572k | 0 + 695M + 1604M | matches BLER, ~52% FLOPs |
| **exp65 (sink + nano + large)** | **467k** | **0 + 320M + 1604M** | **matches BLER, ~40% FLOPs (if it works)** |

## Recipe

Same as exp26: α=2e-3, asym warm-start (warm nano from dense_nano,
cold large), seed 67, 12k steps. Only changes: nano replaces small in the
middle slot, sink replaces nothing (new tier).

## Cluster

`select=1:ncpus=8:ngpus=1:mem=32gb:scratch_ssd=40gb`, walltime 3h.
