"""Export a fixed-size training subset with Array3D features to persistent storage.

Run once on the cluster (submit as a job or run interactively):
    uv run python scripts/export_train_subset.py

Output: two Arrow directories at --output/{uma,tdlc} ready for use with
    training.hf_train_data_dir=<output>
    training.hf_max_samples=<n_samples>
"""

import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Vack0/moe-5g-nrx")
    parser.add_argument("--profiles", nargs="+", default=["uma", "tdlc"])
    parser.add_argument("--n_samples", type=int, default=50_000)
    parser.add_argument(
        "--output",
        default="/storage/brno2/home/ramsay/moe-5g-datasets/train-50k-array3d",
    )
    args = parser.parse_args()

    from datasets import Array3D, Features, Value, load_dataset

    features = Features(
        {
            "inputs": Array3D(dtype="float32", shape=(16, 128, 14)),
            "bit_labels": Array3D(dtype="float32", shape=(4, 128, 14)),
            "channel_target": Array3D(dtype="float32", shape=(8, 128, 14)),
            "snr_db": Value("float32"),
        }
    )

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    for profile in args.profiles:
        out_path = out_root / profile
        if out_path.exists():
            print(f"[SKIP] {profile}: {out_path} already exists")
            continue

        print(f"[{profile}] loading from {args.repo}...", flush=True)
        t0 = time.perf_counter()
        ds = load_dataset(args.repo, profile, split="train")
        print(f"[{profile}] loaded {len(ds)} samples in {time.perf_counter() - t0:.0f}s")

        print(f"[{profile}] selecting {args.n_samples} samples and casting to Array3D...", flush=True)
        t1 = time.perf_counter()
        ds = ds.select(range(args.n_samples)).cast(features)
        print(f"[{profile}] cast done in {time.perf_counter() - t1:.0f}s")

        print(f"[{profile}] saving to {out_path}...", flush=True)
        t2 = time.perf_counter()
        ds.save_to_disk(str(out_path))
        elapsed = time.perf_counter() - t2
        size_gb = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / 1e9
        print(f"[{profile}] saved {len(ds)} samples ({size_gb:.1f} GB) in {elapsed:.0f}s")

    print("\nDone. Set in your job:")
    print(f"  training.hf_train_data_dir={args.output}")
    print(f"  training.hf_max_samples={args.n_samples}")


if __name__ == "__main__":
    main()
