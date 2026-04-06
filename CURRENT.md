# Current Work — 2026-04-06

## What We Learned Today

### Waterfall BLER comparison (waterfall-compare-v1)
- Fine-grained 2 dB SNR sweep: nano/small/large_s32/large_s56 on val data
- **TDLC SNR=17: nano BLER=0.722, large_s56 BLER=0.284 → 44pp gap**
- Average BER/BLER was masking this — all the signal is in the waterfall region
- UMA waterfall is weak (5-8pp gaps), UMA curve barely drops even at SNR=24
- **Expert size matters. MoE direction is justified.**

### state_dim decision: stay at s56
- state_dim=32 was supposed to save stem FLOPs and increase routing leverage
- But stem is SHARED in MoE — state_dim doesn't affect per-expert FLOPs differentiation
- Routing savings: s56=80% vs s32=82% — difference is 2pp, negligible
- Cost of s32: 15pp worse BLER at SNR=17 (large_s32=0.436 vs large_s56=0.284)
- **Verdict: use state_dim=56 throughout. The s32 checkpoints are reference-only.**

### Phase 2 warm-start checkpoints: all ready
- nano (s56, 20k): `model-dense_nano_final20k_constant_lr_s67-aos4hhid:best` ✓
- small (s56, 20k): `model-dense_small_final20k_constant_lr_s67-kivdz4qu:best` ✓
- large (s56, 20k): `model-dense_large_final20k_constant_lr_s67-55l1dpby:best` ✓
- All three use state_dim=56. Phase 2 can start immediately.

### Nano depth: doesn't matter
- 2 blk / 4 blk / 8 blk at nano scale → all within noise
- Keep 4 blocks for nano expert

### GPU utilization: Sionna is the bottleneck
- Profiled on A40: GPU training step = 70ms, Sionna gen (GPU, workers=0) = 483ms
- GPU is idle ~87% of training time
- GPU workers=2 helps somewhat (308ms) but Sionna fights PyTorch for GPU memory (OOM)
- `set_visible_devices` can't be called after TF init → CPU mode needs --force-cpu flag at startup
- CPU profiling results pending (job 18730965)

---

## Currently Running Jobs (MetaCentrum)

| Job ID | Name | Purpose |
|---|---|---|
| `18730965` | profile_dataloader --force-cpu | Measure CPU Sionna gen speed to estimate GPU overlap |
| `18730966` | generate-train-500k-v1 | 500k cached training samples per profile |

---

## Waiting On

1. **CPU profiling (18730965)** — tells us if CPU Sionna + workers achieves meaningful GPU overlap
   - If gen_cpu_w2 < ~150ms → worth wiring up (GPU util >50%)
   - If gen_cpu_w2 > ~300ms → cached training data is the better fix
2. **Training cache (18730966)** — 500k samples/profile to `~/moe-5g-datasets/train-500k-v1/train/`
   - ~24 GB per profile, ~2-3 hours generation time
   - Will need: hook up training to use cached data instead of on-the-fly Sionna

---

## Immediate Next Steps

1. Check CPU profiling results → decide on GPU utilization fix strategy
2. Once training cache is ready → add `cached` training mode (point dataloader at .pt files)
3. **Set up Phase 2 warm-start experiment:**
   - Stage 1: freeze experts, train router ~2-3k steps
   - Stage 2: unfreeze, joint fine-tune ~10k steps (alpha=1e-3, beta=0.1)
   - Evaluate: BLER@high-SNR vs realized FLOPs Pareto curve
4. Update AGENTS.md with Phase 2 experiment artifacts once done

---

## Open Questions

- CPU Sionna speed: fast enough to overlap with GPU training, or not?
- Should training use cached data exclusively, or hybrid (cached + on-the-fly refresh)?
- Is 500k samples enough diversity for 20k training steps (1.28M samples → ~2.5 cycles)?
- Phase 2: does warm-start actually improve over joint-from-scratch (MoE NL Phase 1)?
