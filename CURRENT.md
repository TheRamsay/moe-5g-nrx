# Current Work — 2026-04-06

## What We Learned Today

### Capacity floor (capacity-floor-v1)
- nano (block_dim=8, 4 blk, 90k params) best_score=0.2064 vs small (168k) ~0.202 at 10k steps
- **BER gap is tiny** (~1-2pp) but **BLER gap is real** (~7-8pp TDLC at same steps)
- The BER metric was masking capacity signal — BLER is the right metric going forward
- The waterfall transition width is real but the 7-bin eval spacing was too coarse to see it

### Stem bottleneck (stem-bottleneck-v1)
- state_dim=32 matches large (best_score=0.2016 vs ~0.199) — lossless stem compression
- state_dim=16 degrades: best_score=0.2156, TDLC channel_mse=0.200 vs 0.064 for large (~3×)
- The bottleneck at state_dim=16 is specifically **channel estimation**, not decoding
- Safe working point: state_dim=32, stem=[32,32]

### Revised MoE direction
- Original expert range (block_dim 32/48/64) is too narrow — backbone barely matters in BER
- Wider range needed: **nano/small/large = block_dim 8/32/64**
- Use state_dim=32 throughout — matches quality of state_dim=56, cheaper, makes backbone matter more
- Primary evaluation metric: **BLER at high-SNR bins**, not average BER

---

## Currently Running Jobs (MetaCentrum)

| Job ID | Name | Steps | GPU | Purpose |
|---|---|---|---|---|
| `18723723` | nano_final20k | 20k | 16gb | warm-start checkpoint for redesigned MoE |
| `18723728` | moe_nl_phase1_v1 | 10k | 46gb | Phase 1 with nano/small/large + state_dim=32 |
| `18723729` | large_s32_final20k | 20k | 16gb | warm-start checkpoint for redesigned MoE |
| `18723730` | nano_2blk_depth | 10k | 16gb | does depth matter at nano scale? |
| `18723731` | nano_8blk_depth | 10k | 16gb | does depth matter at nano scale? |

Previously submitted (should be done):
- `18721569` nano 10k (capacity floor)
- `18721570` micro 10k (capacity floor)
- `18721574` stem_s32 10k (stem bottleneck)
- `18721575` stem_s16 10k (stem bottleneck)

---

## Waiting On Before Next Step

1. **nano_final20k** (`18723723`) — need this checkpoint for MoE Phase 2 warm-start
2. **large_s32_final20k** (`18723729`) — need this checkpoint for MoE Phase 2 warm-start
3. **moe_nl_phase1_v1** (`18723728`) — check router entropy + expert utilization to confirm wider range differentiates

Once these three are done, the redesigned MoE Phase 2 (warm-started nano/small/large, state_dim=32) can begin.

---

## Immediate Next Steps (after jobs finish)

1. Check `moe_nl_phase1_v1` routing metrics in W&B:
   - Router entropy should stay high (>0.85) — if not, revisit beta
   - Expert utilization should split meaningfully across SNR bins
   - BLER per SNR bin should show nano being used for easy conditions, large for hard

2. Run evals on nano_final20k and large_s32_final20k checkpoints (same eval script)

3. If MoE NL Phase 1 routing looks good → set up Phase 2 warm-start:
   - Freeze experts, train router ~2-3k steps
   - Unfreeze, joint fine-tune ~10k steps
   - Evaluate on BLER@high-SNR vs realized FLOPs Pareto curve

4. Update AGENTS.md with nano and large_s32 final checkpoint artifacts once evals are done

---

## Open Questions

- Does the MoE NL Phase 1 route differently across SNR bins vs the old small/mid/large MoE?
- Does depth matter for nano (2 vs 4 vs 8 blocks)? Informs final expert design.
- Is BLER@high-SNR the right Pareto axis, or should we use BLER at a specific operating SNR?
- Should small expert also be retrained with state_dim=32 for warm-start, or keep the existing 20k checkpoint?
