#!/usr/bin/env python
"""Evaluate a trained model on test datasets.

Usage:
    # Evaluate on default test dataset
    python scripts/evaluate.py --checkpoint checkpoints/static_dense.pt

    # Evaluate on specific test dataset
    python scripts/evaluate.py --checkpoint checkpoints/model.pt --dataset data/test/uma.pt

    # Evaluate on all test profiles
    python scripts/evaluate.py --checkpoint checkpoints/model.pt --all-profiles

    # Detailed SNR-binned analysis
    python scripts/evaluate.py --checkpoint checkpoints/model.pt --snr-bins 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.data import CachedNRXBatch, build_cached_dataloader
from src.models import StaticDenseNRX

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Build model from saved config
    model = StaticDenseNRX(
        input_channels=int(config["data"]["input_channels"]),
        output_bits=int(config["data"]["bits_per_symbol"])
        * int(config["data"]["num_subcarriers"])
        * int(config["data"]["num_ofdm_symbols"]),
        state_dim=int(config["model"]["backbone"]["state_dim"]),
        num_cnn_blocks=int(config["model"]["backbone"]["num_cnn_blocks"]),
        stem_hidden_dims=list(config["model"]["backbone"]["stem_hidden_dims"]),
        block_hidden_dim=int(config["model"]["backbone"]["block_hidden_dim"]),
        readout_hidden_dim=int(config["model"]["backbone"]["readout_hidden_dim"]),
        activation=str(config["model"]["backbone"]["activation"]),
        num_subcarriers=int(config["data"]["num_subcarriers"]),
        num_ofdm_symbols=int(config["data"]["num_ofdm_symbols"]),
        bits_per_symbol=int(config["data"]["bits_per_symbol"]),
        num_rx_antennas=int(config["data"]["num_rx_antennas"]),
        pilot_symbol_indices=list(config["model"]["backbone"]["pilot_symbol_indices"]),
        pilot_subcarrier_spacing=int(config["model"]["backbone"]["pilot_subcarrier_spacing"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    bits_per_symbol: int,
    snr_bins: int | None = None,
) -> dict[str, Any]:
    """Evaluate model on a dataset.

    Args:
        model: Trained model.
        dataloader: DataLoader for test dataset.
        device: Device to run evaluation on.
        bits_per_symbol: Number of bits per symbol (for SER calculation).
        snr_bins: If provided, compute metrics per SNR bin.

    Returns:
        Dictionary with evaluation metrics.
    """
    loss_fn = torch.nn.BCEWithLogitsLoss()
    channel_loss_fn = torch.nn.MSELoss()

    # Accumulators for overall metrics
    total_loss = 0.0
    total_bit_errors = 0
    total_bits = 0
    total_block_errors = 0
    total_blocks = 0
    total_symbol_errors = 0
    total_symbols = 0
    total_channel_mse = 0.0
    num_batches = 0

    # SNR-binned accumulators
    all_snr_db: list[float] = []
    all_bit_errors: list[int] = []
    all_bits_per_sample: list[int] = []
    all_block_errors: list[int] = []

    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
        batch: CachedNRXBatch
        features = batch.inputs.to(device)
        targets = batch.bit_labels.flatten(start_dim=1).to(device)
        channel_targets = batch.channel_target.to(device)
        snr_db = batch.snr_db

        outputs = model(features)
        logits = outputs["logits"]
        channel_estimate = outputs["channel_estimate"]

        # Loss
        loss = loss_fn(logits, targets)
        total_loss += float(loss.item())

        # Compute per-sample metrics
        predictions = (logits > 0.0).to(dtype=targets.dtype)
        bit_errors = (predictions != targets).to(dtype=torch.float32)

        # Per-sample bit errors (for SNR analysis)
        per_sample_bit_errors = bit_errors.sum(dim=1).cpu().numpy()
        per_sample_bits = targets.shape[1]
        per_sample_block_error = bit_errors.any(dim=1).cpu().numpy().astype(int)

        all_snr_db.extend(snr_db.numpy().tolist())
        all_bit_errors.extend(per_sample_bit_errors.tolist())
        all_bits_per_sample.extend([per_sample_bits] * len(snr_db))
        all_block_errors.extend(per_sample_block_error.tolist())

        # Aggregate metrics
        total_bit_errors += int(bit_errors.sum().item())
        total_bits += int(targets.numel())

        block_errors = bit_errors.any(dim=1).to(dtype=torch.float32)
        total_block_errors += int(block_errors.sum().item())
        total_blocks += int(block_errors.numel())

        symbol_errors = bit_errors.view(bit_errors.shape[0], -1, bits_per_symbol).any(dim=-1).to(dtype=torch.float32)
        total_symbol_errors += int(symbol_errors.sum().item())
        total_symbols += int(symbol_errors.numel())

        channel_mse = channel_loss_fn(channel_estimate, channel_targets)
        total_channel_mse += float(channel_mse.item())

        num_batches += 1

    results = {
        "loss": total_loss / max(num_batches, 1),
        "ber": total_bit_errors / max(total_bits, 1),
        "bler": total_block_errors / max(total_blocks, 1),
        "ser": total_symbol_errors / max(total_symbols, 1),
        "channel_mse": total_channel_mse / max(num_batches, 1),
        "num_samples": total_blocks,
        "num_bit_errors": total_bit_errors,
        "num_block_errors": total_block_errors,
    }

    # SNR-binned analysis
    if snr_bins is not None and snr_bins > 0:
        snr_array = np.array(all_snr_db)
        bit_errors_array = np.array(all_bit_errors)
        bits_per_sample_array = np.array(all_bits_per_sample)
        block_errors_array = np.array(all_block_errors)

        # Create SNR bins
        snr_min, snr_max = snr_array.min(), snr_array.max()
        bin_edges = np.linspace(snr_min, snr_max, snr_bins + 1)
        bin_indices = np.digitize(snr_array, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, snr_bins - 1)

        snr_binned = {}
        for i in range(snr_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue

            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            bin_ber = bit_errors_array[mask].sum() / bits_per_sample_array[mask].sum()
            bin_bler = block_errors_array[mask].mean()

            snr_binned[f"bin_{i}"] = {
                "snr_center": float(bin_center),
                "snr_range": (float(bin_edges[i]), float(bin_edges[i + 1])),
                "ber": float(bin_ber),
                "bler": float(bin_bler),
                "num_samples": int(mask.sum()),
            }

        results["snr_binned"] = snr_binned

    return results


def print_results(results: dict[str, Any], dataset_name: str) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"Results for: {dataset_name}")
    print(f"{'=' * 60}")
    print(f"  Samples:      {results['num_samples']:,}")
    print(f"  Loss:         {results['loss']:.6f}")
    print(f"  BER:          {results['ber']:.6e}")
    print(f"  BLER:         {results['bler']:.4f} ({results['num_block_errors']:,} / {results['num_samples']:,})")
    print(f"  SER:          {results['ser']:.6e}")
    print(f"  Channel MSE:  {results['channel_mse']:.6f}")

    if "snr_binned" in results:
        print("\n  SNR-Binned BLER:")
        print(f"  {'SNR (dB)':>12} | {'BLER':>10} | {'Samples':>10}")
        print(f"  {'-' * 12}-+-{'-' * 10}-+-{'-' * 10}")
        for bin_data in results["snr_binned"].values():
            snr_range = bin_data["snr_range"]
            print(
                f"  {snr_range[0]:5.1f}-{snr_range[1]:5.1f} | {bin_data['bler']:10.4f} | {bin_data['num_samples']:10,}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on test datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to test dataset (default: from config evaluation.dataset_path)",
    )
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="Evaluate on all test profiles (uma, tdlc, mixed)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation (default: 64)",
    )
    parser.add_argument(
        "--snr-bins",
        type=int,
        default=None,
        help="Number of SNR bins for detailed analysis (default: None)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run evaluation on (default: cpu)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "test",
        help="Directory containing test datasets (for --all-profiles)",
    )

    args = parser.parse_args()

    # Resolve device
    device = torch.device(args.device)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    if not args.checkpoint.exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    model, config = load_checkpoint(args.checkpoint, device)
    bits_per_symbol = int(config["data"]["bits_per_symbol"])
    print(f"  Model: {config['model']['name']}")
    print(f"  Global step: {torch.load(args.checkpoint, weights_only=False)['global_step']}")

    # Determine datasets to evaluate
    if args.all_profiles:
        profiles = ["uma", "tdlc", "mixed"]
        datasets = [(p, args.data_dir / f"{p}.pt") for p in profiles]
    elif args.dataset is not None:
        datasets = [(args.dataset.stem, args.dataset)]
    else:
        # Use default from config
        default_path = Path(config.get("evaluation", {}).get("dataset_path", "data/test/mixed.pt"))
        datasets = [(default_path.stem, default_path)]

    # Evaluate each dataset
    all_results = {}
    for name, path in datasets:
        if not path.exists():
            print(f"\n[WARNING] Dataset not found: {path}, skipping")
            continue

        print(f"\nLoading dataset: {path}")
        dataloader = build_cached_dataloader(
            path=path,
            batch_size=args.batch_size,
            shuffle=False,
        )

        results = evaluate_dataset(
            model=model,
            dataloader=dataloader,
            device=device,
            bits_per_symbol=bits_per_symbol,
            snr_bins=args.snr_bins,
        )
        all_results[name] = results
        print_results(results, name)

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("Summary")
        print(f"{'=' * 60}")
        print(f"  {'Profile':>10} | {'BLER':>10} | {'BER':>12} | {'Loss':>10}")
        print(f"  {'-' * 10}-+-{'-' * 10}-+-{'-' * 12}-+-{'-' * 10}")
        for name, results in all_results.items():
            print(f"  {name:>10} | {results['bler']:10.4f} | {results['ber']:12.6e} | {results['loss']:10.6f}")


if __name__ == "__main__":
    main()
