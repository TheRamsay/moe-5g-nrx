"""Convert a HuggingFace Arrow test-set directory to a .pt file for evaluate.py.

Loads data from {arrow_dir}/, gathers (inputs, bit_labels, channel_target,
snr_db) into tensors, saves as {out_path}.pt in the format expected by
src.data.cached_dataset.CachedNRXDataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_from_disk


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--profile", default=None, help="Channel-profile metadata field (e.g., o1_3p5). Defaults to arrow dir name."
    )
    args = parser.parse_args()

    profile = args.profile or args.arrow_dir.name
    print(f"[INFO] Loading {args.arrow_dir} ...", file=sys.stderr)
    ds = load_from_disk(str(args.arrow_dir))
    print(f"[INFO] Loaded {len(ds)} samples; columns = {ds.column_names}", file=sys.stderr)

    # Stack all into tensors
    print("[INFO] Stacking tensors...", file=sys.stderr)
    inputs = torch.tensor(ds["inputs"], dtype=torch.float32)
    bit_labels = torch.tensor(ds["bit_labels"], dtype=torch.float32)
    channel_target = torch.tensor(ds["channel_target"], dtype=torch.float32)
    snr_db = torch.tensor(ds["snr_db"], dtype=torch.float32)

    print(
        f"[INFO] inputs {inputs.shape} | bit_labels {bit_labels.shape} | "
        f"channel_target {channel_target.shape} | snr_db {snr_db.shape}",
        file=sys.stderr,
    )

    bundle = {
        "inputs": inputs,
        "bit_labels": bit_labels,
        "channel_target": channel_target,
        "snr_db": snr_db,
        "metadata": {
            "channel_profile": profile,
            "modulation": "qam16",
            "num_samples": len(ds),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.out)
    size_gb = args.out.stat().st_size / 1e9
    print(f"[INFO] saved {args.out} ({size_gb:.2f} GB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
