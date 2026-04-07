# Current Work — 2026-04-07

## What We Learned Today

### Training cache generation is blocked by memory issues
Five PBS jobs have failed attempting to generate 500k training samples per profile.
See `PROBLEM.md` for the full crash history. TL;DR:

1. **TF session memory leak** — single TF session for 500k samples stalls mid-run (fixed by subprocess chunking)
2. **PyTorch allocator doesn't return freed pages to OS** — parent RSS grows by ~4.8 GB per completed chunk even after `del shard; gc.collect()`. By chunk 3, parent (~34 GB) + child TF CPU (~25 GB) > 64 GB cgroup limit → OOM.

**Proposed fix (coded, not yet submitted):** call `ctypes.CDLL("libc.so.6").malloc_trim(0)` after each shard deletion in the parent. Forces glibc to return freed heap pages to the OS, keeping parent RSS stable at ~25 GB. Expected peak: ~50 GB → safe under 64 GB limit.

**Current code state:** fix is in `scripts/generate_datasets.py` (malloc_trim + subprocess isolation + CPU-only TF). Job not yet resubmitted.

---

## Currently Running Jobs

None.

---

## Immediate Next Steps

1. **Resubmit training cache generation** with malloc_trim fix
2. Wire up `cached` training mode once `train-500k-v1` is available
3. **Set up Phase 2 warm-start experiment:**
   - Stage 1: freeze experts, train router ~2-3k steps
   - Stage 2: unfreeze, joint fine-tune ~10k steps (alpha=1e-3, beta=0.1)
   - Evaluate: BLER@high-SNR vs realized FLOPs Pareto curve

---

## Open Questions

- Does malloc_trim fully solve the RSS accumulation, or do we need 96/128 GB?
- Should training use cached data exclusively, or hybrid (cached + on-the-fly refresh)?
- Is 500k samples enough diversity for 20k training steps (~2.5 epochs)?
- Phase 2: does warm-start actually improve over joint-from-scratch (Phase 1)?
