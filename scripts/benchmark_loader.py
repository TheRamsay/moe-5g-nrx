"""Benchmark data loading for mixed-profile training (HF hub source).

Historical: compared alternating vs chunked profile iteration. Kept for
reproducibility; the training pipeline itself now preloads into RAM and
runs with num_workers=0 (see src/data/train_loader.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

from datasets import load_dataset
from torch.utils.data import DataLoader

from src.data.cached_dataset import PreloadedBatchedTrainDataset

HF_REPO = "Vack0/moe-5g-nrx"
PROFILES = ["uma", "tdlc"]
BATCH_SIZE = 128
SEED = 67


def _identity_collate(batch):
    return batch[0] if isinstance(batch, list) else batch


def make_loader(profile, num_workers=4, prefetch=1):
    arrow = load_dataset(HF_REPO, profile, split="train").with_format("torch")
    ds = PreloadedBatchedTrainDataset(
        arrow, profile=profile, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, base_seed=SEED
    )
    return DataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_identity_collate,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch if num_workers > 0 else None,
    )


def benchmark_alternating(n_steps=200, num_workers=4, prefetch=1):
    """Current approach: two loaders, alternate every batch."""
    loaders = {p: make_loader(p, num_workers, prefetch) for p in PROFILES}
    iters = {p: iter(dl) for p, dl in loaders.items()}

    times = []
    for i in range(n_steps):
        profile = PROFILES[i % len(PROFILES)]
        t0 = time.perf_counter()
        try:
            batch = next(iters[profile])
        except StopIteration:
            iters[profile] = iter(loaders[profile])
            batch = next(iters[profile])
        _ = batch.inputs.cuda()
        times.append(time.perf_counter() - t0)
    return times


def benchmark_chunked(n_steps=200, chunk_size=10, num_workers=4, prefetch=1):
    """Alternate every N batches instead of every batch."""
    loaders = {p: make_loader(p, num_workers, prefetch) for p in PROFILES}
    iters = {p: iter(dl) for p, dl in loaders.items()}

    times = []
    for i in range(n_steps):
        profile = PROFILES[(i // chunk_size) % len(PROFILES)]
        t0 = time.perf_counter()
        try:
            batch = next(iters[profile])
        except StopIteration:
            iters[profile] = iter(loaders[profile])
            batch = next(iters[profile])
        _ = batch.inputs.cuda()
        times.append(time.perf_counter() - t0)
    return times


def report(name, times):
    """Print timing statistics."""
    import numpy as np

    t = np.array(times[20:])  # skip warmup
    p50 = np.median(t)
    p95 = np.percentile(t, 95)
    print(f"{name:40s} | mean={t.mean():.4f}s  p50={p50:.4f}s  p95={p95:.4f}s  steps/s={1 / t.mean():.1f}")


if __name__ == "__main__":
    print("Benchmarking data loading strategies (200 steps, data load only, no model)...")
    print("=" * 95)

    times = benchmark_alternating(n_steps=200, num_workers=4, prefetch=1)
    report("alt w4 p1 (current)", times)

    times = benchmark_alternating(n_steps=200, num_workers=4, prefetch=2)
    report("alt w4 p2", times)

    times = benchmark_alternating(n_steps=200, num_workers=4, prefetch=4)
    report("alt w4 p4", times)

    times = benchmark_chunked(n_steps=200, chunk_size=10, num_workers=4, prefetch=1)
    report("chunk=10 w4 p1", times)

    times = benchmark_chunked(n_steps=200, chunk_size=50, num_workers=4, prefetch=1)
    report("chunk=50 w4 p1", times)

    times = benchmark_chunked(n_steps=200, chunk_size=10, num_workers=4, prefetch=2)
    report("chunk=10 w4 p2", times)

    print("=" * 95)
    print("Done!")
