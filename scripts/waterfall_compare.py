#!/usr/bin/env python
"""Compare BLER waterfall curves across checkpoints with fine-grained SNR binning.

Usage:
    # From W&B artifacts (downloads automatically):
    python scripts/waterfall_compare.py \
        --artifacts \
            "knn_moe-5g-nrx/moe-5g-nrx/model-dense_nano_final20k_constant_lr_s67-aos4hhid:best" \
            "knn_moe-5g-nrx/moe-5g-nrx/model-dense_large_s32_final20k_constant_lr_s67-rdfefyt1:best" \
        --labels nano large_s32 \
        --data-dir data/val \
        --profiles tdlc uma \
        --snr-step 2

    # From local checkpoint files:
    python scripts/waterfall_compare.py \
        --checkpoints path/to/nano.pt path/to/large.pt \
        --labels nano large_s32 \
        --data-dir data/val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CachedNRXBatch, build_cached_dataloader  # noqa: E402
from src.models import build_model_from_config  # noqa: E402
from src.utils.metrics import compute_batch_error_metrics  # noqa: E402


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def download_artifact(artifact_ref: str) -> Path:
    import wandb

    artifact = wandb.Api().artifact(artifact_ref)
    download_dir = Path(artifact.download())
    files = [p for p in download_dir.iterdir() if p.is_file()]
    if len(files) == 1:
        return files[0]
    raise FileNotFoundError(f"Expected 1 file in {download_dir}, got {len(files)}")


@torch.no_grad()
def collect_per_sample(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    bits_per_symbol: int,
) -> dict[str, np.ndarray]:
    """Run model on all data, return per-sample SNR, bit errors, block errors."""
    all_snr = []
    all_bit_errors = []
    all_block_errors = []
    all_bits = []

    for batch in tqdm(dataloader, desc="  inference", leave=False):
        batch: CachedNRXBatch
        features = batch.inputs.to(device)
        targets = batch.bit_labels.flatten(start_dim=1).to(device)
        snr_db = batch.snr_db

        outputs = model(features)
        logits = outputs["logits"]

        metrics = compute_batch_error_metrics(
            logits,
            targets,
            bits_per_symbol=bits_per_symbol,
            num_subcarriers=features.shape[2],
            num_ofdm_symbols=features.shape[3],
        )

        all_snr.append(snr_db.numpy())
        all_bit_errors.append(metrics.bit_errors.sum(dim=1).cpu().numpy())
        all_block_errors.append(metrics.block_errors.cpu().numpy())
        all_bits.append(np.full(len(snr_db), targets.shape[1]))

    return {
        "snr": np.concatenate(all_snr),
        "bit_errors": np.concatenate(all_bit_errors),
        "block_errors": np.concatenate(all_block_errors),
        "bits": np.concatenate(all_bits),
    }


def bin_metrics(
    data: dict[str, np.ndarray],
    bin_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute BER and BLER per SNR bin."""
    snr = data["snr"]
    n_bins = len(bin_edges) - 1
    bin_idx = np.digitize(snr, bin_edges[:-1]) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    centers = np.zeros(n_bins)
    ber = np.full(n_bins, np.nan)
    bler = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = bin_idx == i
        n = mask.sum()
        counts[i] = n
        centers[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
        if n > 0:
            ber[i] = data["bit_errors"][mask].sum() / data["bits"][mask].sum()
            bler[i] = data["block_errors"][mask].mean()

    return {"centers": centers, "ber": ber, "bler": bler, "counts": counts}


def main():
    parser = argparse.ArgumentParser(description="Waterfall BLER comparison across checkpoints")
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--checkpoints", nargs="+", type=str, help="Local checkpoint paths")
    ckpt_group.add_argument("--artifacts", nargs="+", type=str, help="W&B artifact references")
    parser.add_argument("--labels", nargs="+", type=str, required=True, help="Label for each checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/val", help="Directory with {profile}.pt files")
    parser.add_argument("--profiles", nargs="+", default=["tdlc", "uma"], help="Channel profiles to evaluate")
    parser.add_argument("--snr-step", type=float, default=2.0, help="SNR bin width in dB (default: 2)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="waterfall_compare.png", help="Output plot filename")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Resolve checkpoints
    if args.artifacts:
        ckpt_paths = [download_artifact(a) for a in args.artifacts]
    else:
        ckpt_paths = [Path(p) for p in args.checkpoints]

    if len(args.labels) != len(ckpt_paths):
        parser.error(f"Got {len(ckpt_paths)} checkpoints but {len(args.labels)} labels")

    # Load models
    models = {}
    for label, path in zip(args.labels, ckpt_paths):
        print(f"Loading {label} from {path}")
        model, config = load_checkpoint(path, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {n_params:,} params")
        models[label] = model

    # Resolve data directory
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    # SNR ranges per profile (must match dataset generation)
    snr_ranges = {
        "tdlc": (-10.0, 20.0),
        "uma": (-5.0, 25.0),
    }

    # Evaluate each model on each profile
    # results[label][profile] = binned metrics dict
    results: dict[str, dict[str, dict]] = {label: {} for label in args.labels}

    for profile in args.profiles:
        dataset_path = data_dir / f"{profile}.pt"
        if not dataset_path.exists():
            print(f"WARNING: {dataset_path} not found, skipping {profile}")
            continue

        print(f"\n--- {profile.upper()} ({dataset_path}) ---")
        dataloader = build_cached_dataloader(dataset_path, batch_size=args.batch_size)

        snr_min, snr_max = snr_ranges.get(profile, (-10, 25))
        bin_edges = np.arange(snr_min, snr_max + args.snr_step, args.snr_step)

        for label in args.labels:
            print(f"  Evaluating {label}...")
            data = collect_per_sample(models[label], dataloader, device, bits_per_symbol=4)
            binned = bin_metrics(data, bin_edges)
            results[label][profile] = binned

    # Print comparison table
    for profile in args.profiles:
        if not any(profile in results[lbl] for lbl in args.labels):
            continue
        print(f"\n{'=' * 70}")
        print(f"  {profile.upper()} — BLER Waterfall Comparison")
        print(f"{'=' * 70}")

        # Header
        header = f"  {'SNR (dB)':>10}"
        for label in args.labels:
            header += f" | {label + ' BLER':>12}"
        header += " |    delta"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")

        ref = results[args.labels[0]].get(profile)
        if ref is None:
            continue

        for i, center in enumerate(ref["centers"]):
            row = f"  {center:10.1f}"
            vals = []
            for label in args.labels:
                binned = results[label].get(profile, {})
                bler = binned.get("bler", np.array([]))[i] if binned else np.nan
                vals.append(bler)
                if np.isnan(bler):
                    row += f" | {'n/a':>12}"
                else:
                    row += f" | {bler:12.4f}"
            # Delta between first and last
            if len(vals) >= 2 and not np.isnan(vals[0]) and not np.isnan(vals[-1]):
                delta = vals[0] - vals[-1]
                row += f" | {delta:+8.4f}"
            else:
                row += f" | {'':>8}"
            print(row)

    # Plot
    n_profiles = sum(1 for p in args.profiles if any(p in results[lbl] for lbl in args.labels))
    if n_profiles == 0:
        print("\nNo data to plot.")
        return

    fig, axes = plt.subplots(1, n_profiles, figsize=(7 * n_profiles, 5), squeeze=False)
    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v"]

    plot_idx = 0
    for profile in args.profiles:
        if not any(profile in results[lbl] for lbl in args.labels):
            continue
        ax = axes[0, plot_idx]
        for j, label in enumerate(args.labels):
            binned = results[label].get(profile)
            if binned is None:
                continue
            valid = ~np.isnan(binned["bler"])
            ax.semilogy(
                binned["centers"][valid],
                binned["bler"][valid],
                marker=markers[j % len(markers)],
                color=colors[j % len(colors)],
                label=label,
                linewidth=2,
                markersize=5,
            )

        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("BLER")
        ax.set_title(f"{profile.upper()} Waterfall")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=1e-3, top=1.5)
        plot_idx += 1

    fig.tight_layout()
    output_path = PROJECT_ROOT / args.output
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
