"""Benchmark different data loading strategies for mixed-profile training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

from src.data import build_cached_dataloader


def benchmark_alternating(n_steps=200, batch_size=128, num_workers=4, prefetch=1):
    """Current approach: two separate loaders, alternate every batch."""
    uma_loader = build_cached_dataloader(
        hf_repo="Vack0/moe-5g-nrx",
        hf_config="uma",
        hf_split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=prefetch,
    )
    tdlc_loader = build_cached_dataloader(
        hf_repo="Vack0/moe-5g-nrx",
        hf_config="tdlc",
        hf_split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=prefetch,
    )
    uma_iter = iter(uma_loader)
    tdlc_iter = iter(tdlc_loader)

    times = []
    for i in range(n_steps):
        t0 = time.perf_counter()
        try:
            batch = next(uma_iter if i % 2 == 0 else tdlc_iter)
        except StopIteration:
            if i % 2 == 0:
                uma_iter = iter(uma_loader)
                batch = next(uma_iter)
            else:
                tdlc_iter = iter(tdlc_loader)
                batch = next(tdlc_iter)
        # Simulate GPU transfer
        _ = batch[0].cuda()
        times.append(time.perf_counter() - t0)

    return times


def benchmark_alternating_chunked(n_steps=200, chunk_size=10, batch_size=128, num_workers=4, prefetch=1):
    """Alternate every N batches instead of every batch."""
    uma_loader = build_cached_dataloader(
        hf_repo="Vack0/moe-5g-nrx",
        hf_config="uma",
        hf_split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=prefetch,
    )
    tdlc_loader = build_cached_dataloader(
        hf_repo="Vack0/moe-5g-nrx",
        hf_config="tdlc",
        hf_split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=prefetch,
    )
    uma_iter = iter(uma_loader)
    tdlc_iter = iter(tdlc_loader)

    times = []
    for i in range(n_steps):
        t0 = time.perf_counter()
        use_uma = (i // chunk_size) % 2 == 0
        try:
            batch = next(uma_iter if use_uma else tdlc_iter)
        except StopIteration:
            if use_uma:
                uma_iter = iter(uma_loader)
                batch = next(uma_iter)
            else:
                tdlc_iter = iter(tdlc_loader)
                batch = next(tdlc_iter)
        _ = batch[0].cuda()
        times.append(time.perf_counter() - t0)

    return times


def benchmark_single_interleaved(n_steps=200, batch_size=128, num_workers=8, prefetch=1):
    """Single loader with interleaved HF dataset (both profiles merged)."""
    from datasets import concatenate_datasets, load_dataset

    uma_ds = load_dataset("Vack0/moe-5g-nrx", "uma", split="train")
    tdlc_ds = load_dataset("Vack0/moe-5g-nrx", "tdlc", split="train")

    # Add profile column
    uma_ds = uma_ds.add_column("profile", ["uma"] * len(uma_ds))
    tdlc_ds = tdlc_ds.add_column("profile", ["tdlc"] * len(tdlc_ds))
    merged = concatenate_datasets([uma_ds, tdlc_ds]).shuffle(seed=67)

    # Use the same batch-native path
    loader = build_cached_dataloader(
        hf_dataset=merged,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=prefetch,
    )

    times = []
    data_iter = iter(loader)
    for i in range(n_steps):
        t0 = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        _ = batch[0].cuda()
        times.append(time.perf_counter() - t0)

    return times


def report(name, times):
    """Print timing statistics."""
    import numpy as np

    t = np.array(times[20:])  # skip warmup
    p50 = np.median(t)
    p95 = np.percentile(t, 95)
    print(f"{name:40s} | mean={t.mean():.4f}s | p50={p50:.4f}s | p95={p95:.4f}s | steps/s={1/t.mean():.1f}")


if __name__ == "__main__":
    print("Benchmarking data loading strategies (200 steps each)...")
    print("=" * 100)

    # Current approach
    times = benchmark_alternating(n_steps=200, num_workers=4, prefetch=1)
    report("alternating w4 p1 (current)", times)

    # More prefetch
    times = benchmark_alternating(n_steps=200, num_workers=4, prefetch=2)
    report("alternating w4 p2", times)

    times = benchmark_alternating(n_steps=200, num_workers=4, prefetch=4)
    report("alternating w4 p4", times)

    # Chunked alternation
    times = benchmark_alternating_chunked(n_steps=200, chunk_size=10, num_workers=4, prefetch=1)
    report("alternating chunk=10 w4 p1", times)

    times = benchmark_alternating_chunked(n_steps=200, chunk_size=50, num_workers=4, prefetch=1)
    report("alternating chunk=50 w4 p1", times)

    print("=" * 100)
    print("Done!")
