"""Split an existing Array3D training dataset into per-worker shards.

Each shard is loaded independently by one DataLoader worker — no shared
parent memory, no CoW blowup, stable RAM across all epochs.

Run once on the cluster (interactive or as a job):
    uv run python scripts/export_train_shards.py

Output layout (matches hf_num_workers=2):
    <output>/{uma,tdlc}/shard-0/   (~25k samples, ~5GB)
    <output>/{uma,tdlc}/shard-1/   (~25k samples, ~5GB)

Use in training jobs:
    training.hf_train_data_dir=<output>
    training.hf_max_samples=50000
"""

import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default="/storage/brno2/home/ramsay/moe-5g-datasets/train-50k-array3d",
        help="Source Array3D dataset directory (with uma/ and tdlc/ subdirs)",
    )
    parser.add_argument(
        "--output",
        default="/storage/brno2/home/ramsay/moe-5g-datasets/train-50k-sharded",
    )
    parser.add_argument("--profiles", nargs="+", default=["uma", "tdlc"])
    parser.add_argument("--num_shards", type=int, default=2, help="Must match hf_num_workers")
    args = parser.parse_args()

    from datasets import load_from_disk

    src = Path(args.src)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for profile in args.profiles:
        src_path = src / profile
        if not src_path.exists():
            print(f"[SKIP] {profile}: source not found at {src_path}")
            continue

        print(f"[{profile}] loading from {src_path}...", flush=True)
        t0 = time.perf_counter()
        ds = load_from_disk(str(src_path))
        n = len(ds)
        print(f"[{profile}] loaded {n} samples in {time.perf_counter() - t0:.0f}s")

        shard_size = n // args.num_shards
        for i in range(args.num_shards):
            shard_out = out / profile / f"shard-{i}"
            if shard_out.exists():
                print(f"[SKIP] {profile}/shard-{i}: already exists")
                continue
            start = i * shard_size
            end = n if i == args.num_shards - 1 else (i + 1) * shard_size
            print(f"[{profile}] shard-{i}: rows {start}-{end} ({end - start} samples)...", flush=True)
            t1 = time.perf_counter()
            shard_out.parent.mkdir(parents=True, exist_ok=True)
            ds.select(range(start, end)).save_to_disk(str(shard_out))
            gb = sum(f.stat().st_size for f in shard_out.rglob("*") if f.is_file()) / 1e9
            print(f"[{profile}] shard-{i}: saved {gb:.1f} GB in {time.perf_counter() - t1:.0f}s")

    print("\nDone. Use in training jobs:")
    print(f"  training.hf_train_data_dir={args.output}")
    print("  training.hf_max_samples=50000")
    print(f"  training.hf_num_workers={args.num_shards}  # must match num_shards")


if __name__ == "__main__":
    main()
