# Current Work — 2026-04-11

## What We Learned Today

### HuggingFace dataset is solid — training bottleneck unblocked
A friend uploaded the 5G NRX dataset to `Vack0/moe-5g-nrx` on HuggingFace with
250k train + 20k val + 40k test samples per profile (uma, tdlc). Column schema
already matches the project (`inputs`, `bit_labels`, `channel_target`, `snr_db`).

**Replication run** (job 18856089): dense large, 10k steps, mixed uma/tdlc,
trained on HF data instead of on-the-fly Sionna. Hit `best_score=0.1998`
with val tdlc BER=0.1249 and val uma BER=0.2748 — indistinguishable from the
Sionna baseline at the same step count. **The HF dataset produces the same
results as Sionna.** See run `2qgunl39`.

This unblocks everything: we can retire `scripts/generate_datasets.py` and its
malloc_trim workarounds. Phase 2 warm-start can start immediately.

### HF cache must be forced to persistent storage
MetaCentrum pre-sets `HF_HUB_CACHE` and `HF_DATASETS_CACHE` to `$SCRATCHDIR` as
part of the job environment. We learned this the hard way — first training run
re-downloaded ~100GB of parquet to scratch despite setting `HF_HOME` correctly.
Fix in `scripts/metacentrum_job.sh`: `unset` both vars before re-exporting with
unconditional assignment (not `${VAR:-default}` defaulting).

Dataset is now pre-cached at `/storage/brno2/home/ramsay/.cache/huggingface/`
(124GB hub parquets + 126GB Arrow caches for both profiles). Future jobs find
the cache and start training without re-downloading.

### Code changes landed
- `src/data/cached_dataset.py`: new `HuggingFaceNRXDataset` (lazy, memory-mapped
  via Arrow). Also dropped the `metadata` key requirement from `CachedNRXDataset`
  — channel_profile is inferred from filename/config name, modulation defaults
  to `qam16`.
- `main.py`: `_build_hf_train_loader` with `_AlternatingLoader` for mixed
  training. `training.hf_dataset=Vack0/moe-5g-nrx` activates HF-backed training,
  replacing the Sionna iterable pipeline.
- `validation.hf_dataset` also supported but unused so far — existing `.pt`
  val/test files still work.
- `datasets>=3.0.0` added to dependencies.
- `scripts/predownload_hf.sh`: PBS pre-download script (CPU-only, no GPU) for
  warming the HF cache.

### HF loader defaults are now locked
Short MetaCentrum sweeps identified the safe mixed-training setting for the HF
path:

- `training.hf_num_workers=2`
- `training.hf_prefetch_factor=1`

Why this is locked:
- `mixed` training builds two per-profile DataLoaders, so `hf_num_workers=2`
  means 4 workers total.
- The original `4/4` setting overcommitted host resources and left the GPU
  mostly idle.
- Broad sweep winner was initially ambiguous because runs landed on different
  GPU families.
- Controlled confirmation on the same `16 GB` GPU class (`Quadro RTX 5000`)
  showed `w2_p1` beating `w2_p2` clearly.

**Controlled same-GPU confirm runs:**

| Setting | Job | GPU | Walltime | Peak RAM | Final loss | Final BER |
|---|---:|---|---:|---:|---:|---:|
| `w2_p1` (`workers=2`, `prefetch=1`) | `18890316` | Quadro RTX 5000 | **18m10s** | **20.2 GB** | 0.2823 | 0.1404 |
| `w2_p2` (`workers=2`, `prefetch=2`) | `18895412` | Quadro RTX 5000 | 29m22s | 35.8 GB | 0.2720 | 0.1363 |

Interpretation: `prefetch=2` increased host-side in-flight work and memory
pressure without improving throughput. Tiny final-loss differences are normal
run noise; for pipeline tuning, walltime and RAM are the deciding metrics.

---

## Currently Running Jobs

None.

---

## Immediate Next Steps

1. **Phase 2 warm-start** — all three dense checkpoints are finalized; router
   training can start as soon as HF training path is proven stable.
   - Stage 1: freeze experts, train router ~2-3k steps
   - Stage 2: unfreeze, joint fine-tune ~10k steps (alpha=1e-3, beta=0.1)
2. **Interactive loader profiling** — confirm whether the remaining bottleneck
   is per-sample Arrow→torch conversion / Python collation rather than worker
   count.
3. **Retire `generate_datasets.py`** — no longer needed for training data.

---

## Open Questions

- Is per-sample Arrow→torch conversion the actual GPU bottleneck, or is it
  something else? Profile the data pipeline before assuming.
- Should val/test also move to HF dataset, or keep the existing `.pt` files?
  (No urgency — they work.)
- For final comparisons, should the canonical dense baseline be refreshed under
  the same HF train regime as Phase 2 MoE, to remove the train-source confound?
- Phase 2: does warm-start actually improve over joint-from-scratch (Phase 1)?
