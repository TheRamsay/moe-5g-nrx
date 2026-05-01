# Dataset Generation — Crash History & Root Cause Analysis

## Goal

Generate 500k training samples per profile (uma + tdlc) to
`~/moe-5g-datasets/train-500k-v1/train/{uma,tdlc}.pt`.

---

## Attempt 1 — Original script (job 18730966, 32 GB RAM)

**What happened:** Stalled at 158k/500k samples. Rate dropped from ~97 samples/s to 62s/sample.

**Root cause:** TF/Sionna session runs for the entire 500k generation in one go. TF's internal allocator accumulates memory across batches and never releases it. At ~158k samples, the process RSS exceeded 32 GB and Linux began swapping → effectively hung.

**Fix tried:** Bump RAM to 64 GB, pre-allocate output tensor (avoid 2× torch.cat peak).

---

## Attempt 2 — Pre-allocated tensor (job 18734067, 64 GB RAM)

**What happened:** Stalled at 329k/500k (~66%). Rate dropped from ~73 samples/s to 24s/sample.

**Root cause:** Same TF leak, just got further with more RAM. The pre-allocation fix only eliminated the torch.cat 2× peak — TF's session-level memory growth was untouched. One continuous TF session for 7800+ batches keeps accumulating.

**Fix tried:** Chunked generation — split 500k into 5×100k chunks, rebuild dataloader per chunk, `del dataloader; gc.collect()` between chunks.

---

## Attempt 3 — Chunked TF sessions, same process (job 18741832, 64 GB RAM)

**What happened:** Stalled mid-chunk 4 (rate dropped from 63 samples/s to 2s/sample).

**Root cause:** `del dataloader; gc.collect()` does NOT free TF's internal C++ allocator. TF holds its memory pool at the OS level even after Python objects are deleted. Memory grows chunk-over-chunk within the same process. By chunk 4, cumulative TF RSS + parent pre-alloc > 64 GB.

**Fix tried:** True subprocess isolation — each chunk runs in a fresh `spawn` subprocess so TF is fully initialized and fully destroyed (OS reclaims all memory on exit).

---

## Attempt 4 — Subprocess-isolated chunks, GPU TF (job 18747301, 64 GB RAM)

**What happened:** OOM at chunk 3, during TF initialization.

**Root cause:** The `_chunk_worker` subprocess didn't set `CUDA_VISIBLE_DEVICES=""`, so TF claimed the GPU and allocated large host-pinned memory buffers. Combined with the parent's 24 GB pre-allocated tensor and PyTorch's allocator not returning freed shard memory to the OS:

- Parent RSS after 2 shards: ~24 GB (pre-alloc) + ~9.6 GB (freed shards stuck in glibc heap) = ~34 GB
- Child 3 TF GPU init: ~25–30 GB
- Total: ~60–64 GB → OOM

**Fix tried:** Force `CUDA_VISIBLE_DEVICES=""` in worker (CPU-only TF) + pre-allocate in child too.

---

## Attempt 5 — CPU-only TF in workers (job 18755623, 64 GB RAM)

**What happened:** OOM at chunk 3, again at TF initialization (immediately).

**Root cause:** CPU-only TF confirmed working (no GPU device lines in log). But the parent RSS still grows because **glibc does not return freed heap pages to the OS**. After loading and deleting each 4.8 GB shard:

- PyTorch's CPU allocator returns memory to glibc's heap
- glibc does NOT call `madvise(MADV_FREE)` or `sbrk(-n)` automatically
- Parent RSS grows by ~4.8 GB per completed chunk even though tensors are gone

By chunk 3 startup:
- Parent RSS: ~24 GB (pre-alloc) + ~9.6 GB (2 orphaned shard allocations) = ~34 GB  
- Child TF CPU init: ~25 GB  
- Total: ~59 GB → at the 64 GB cgroup limit → OOM

---

## Proposed Fix (not yet run)

Add `ctypes.CDLL("libc.so.6").malloc_trim(0)` after each shard deletion in the parent process. This forces glibc to compact its heap and return free pages to the OS, keeping parent RSS at ~24–25 GB constant across all chunks.

Expected peak at chunk N startup:
- Parent: ~25 GB (stable)
- Child TF CPU: ~25 GB
- Total: ~50 GB → safe under 64 GB

**Status: code fix ready, not yet submitted.**
