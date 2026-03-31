"""
Visualization script for generated data validation.
Usage:
    python scripts/visualize_data.py
    python scripts/visualize_data.py --datasets awgn uma tdlc
    python scripts/visualize_data.py --datasets awgn --batch-size 32 --batches 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))


def _build_cfg(dataset_name: str, batch_size: int, seed: int) -> DictConfig:
    conf_dir = str(PROJECT_ROOT / "conf")
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"dataset={dataset_name}",
                f"training.batch_size={batch_size}",
                f"runtime.seed={seed}",
                "dataset.diagnostics.sample_preview=false",
            ],
        )
    return cfg


def _extract_signal_and_estimates(batch_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract signal and LS estimates from batch inputs.
    Expected channel order in 16-channel tensor:
    - Channels 0-7: Received signal (4 antennas × 2 for re/im)
    - Channels 8-15: LS estimates (4 antennas × 2 for re/im)
    """
    signal = batch_inputs[:, :8, :, :]  # First 8 channels: received signal
    estimates = batch_inputs[:, 8:, :, :]  # Last 8 channels: LS estimates
    return signal, estimates


def _compute_power(tensor: torch.Tensor) -> torch.Tensor:
    """Compute power (sum of squares) across re/im pairs."""
    # Reshape to (..., pairs of 2)
    return (tensor**2).mean(dim=1)  # Average across antenna pairs


def _compute_statistics(batch: object) -> dict:
    """Compute basic statistics from a batch."""
    inputs = batch.inputs
    snr_db = batch.snr_db

    signal, estimates = _extract_signal_and_estimates(inputs)

    return {
        "snr_mean": float(torch.mean(snr_db).item()),
        "snr_std": float(torch.std(snr_db).item()),
        "snr_min": float(torch.min(snr_db).item()),
        "snr_max": float(torch.max(snr_db).item()),
        "signal_power_mean": float(torch.mean(_compute_power(signal)).item()),
        "signal_power_std": float(torch.std(_compute_power(signal)).item()),
        "estimates_power_mean": float(torch.mean(_compute_power(estimates)).item()),
        "estimates_power_std": float(torch.std(_compute_power(estimates)).item()),
    }


def main() -> None:
    from src.data import build_dataloader

    parser = argparse.ArgumentParser(description="Visualize and compare different dataset configurations")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["uma", "tdlc"],
        help="Dataset profiles to visualize (uma/tdlc/mixed)",
    )
    parser.add_argument("--batches", type=int, default=3, help="Number of batches per dataset")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for each dataset")
    parser.add_argument("--seed", type=int, default=67, help="RNG seed")
    parser.add_argument("--output", type=str, default=None, help="Save plots to file instead of displaying")
    args = parser.parse_args()

    # Load data from all datasets
    all_data = {}
    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Loading dataset: {dataset_name}")
        print(f"{'='*60}")

        cfg = _build_cfg(dataset_name, args.batch_size, args.seed)
        dataloader = build_dataloader(cfg)

        batches_data = []
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.batches:
                break
            stats = _compute_statistics(batch)
            batches_data.append(
                {
                    "batch_inputs": batch.inputs,
                    "batch_bits": batch.bit_labels,
                    "batch_snr": batch.snr_db,
                    "stats": stats,
                }
            )
            print(
                f"  Batch {batch_idx}: SNR=[{stats['snr_min']:.1f}, {stats['snr_max']:.1f}] dB, "
                f"Signal Power={stats['signal_power_mean']:.6f}"
            )

        all_data[dataset_name] = batches_data

    # Create comparison plots
    colors = plt.cm.tab10(np.linspace(0, 1, len(args.datasets)))

    # 1. SNR Distribution Comparison
    ax1 = plt.subplot(3, 3, 1)
    for (dataset_name, data), color in zip(all_data.items(), colors):
        snr_values = []
        for batch_data in data:
            snr_values.extend(batch_data["batch_snr"].numpy().flatten())
        ax1.hist(snr_values, bins=20, alpha=0.6, label=dataset_name, color=color)
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("Count")
    ax1.set_title("SNR Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. SNR Box Plot
    ax2 = plt.subplot(3, 3, 2)
    snr_data_for_box = []
    labels_for_box = []
    for dataset_name in args.datasets:
        snr_values = []
        for batch_data in all_data[dataset_name]:
            snr_values.extend(batch_data["batch_snr"].numpy().flatten())
        snr_data_for_box.append(snr_values)
        labels_for_box.append(dataset_name)
    ax2.boxplot(snr_data_for_box, tick_labels=labels_for_box)
    ax2.set_ylabel("SNR (dB)")
    ax2.set_title("SNR Box Plot")
    ax2.grid(True, alpha=0.3)

    # 3. Signal Power Distribution
    ax3 = plt.subplot(3, 3, 3)
    for (dataset_name, data), color in zip(all_data.items(), colors):
        power_values = []
        for batch_data in data:
            signal, _ = _extract_signal_and_estimates(batch_data["batch_inputs"])
            power_values.extend(_compute_power(signal).numpy().flatten())
        ax3.hist(power_values, bins=30, alpha=0.6, label=dataset_name, color=color)
    ax3.set_xlabel("Signal Power")
    ax3.set_ylabel("Count")
    ax3.set_title("Signal Power Distribution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(3, 3, 4)
    for dataset_name in args.datasets:
        # 1. Zoberieme dáta len z PRVÉHO batchu
        first_batch = all_data[dataset_name][0]

        # 2. Vytiahneme ODHAD KANÁLA ('estimates')
        _, estimates = _extract_signal_and_estimates(first_batch["batch_inputs"])

        # 3. Vypočítame výkon. Výstup má pravdepodobne tvar [batch, freq, time]
        power_grid = _compute_power(estimates)

        # 4. Zoberieme len ukážku 0 z daného batchu!
        single_grid = power_grid[0]

        im = ax4.imshow(single_grid.numpy(), cmap="viridis", aspect="auto")
        ax4.set_title(f"{dataset_name} - Channel Estimate (Sample 0)")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Frequency")
        plt.colorbar(im, ax=ax4)

    # 5. LS Estimates Power Distribution
    ax5 = plt.subplot(3, 3, 5)
    for (dataset_name, data), color in zip(all_data.items(), colors):
        est_power_values = []
        for batch_data in data:
            _, estimates = _extract_signal_and_estimates(batch_data["batch_inputs"])
            est_power_values.extend(_compute_power(estimates).numpy().flatten())
        ax5.hist(est_power_values, bins=30, alpha=0.6, label=dataset_name, color=color)
    ax5.set_xlabel("LS Estimates Power")
    ax5.set_ylabel("Count")
    ax5.set_title("LS Estimates Power Distribution")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Signal vs Estimates Power Scatter
    ax6 = plt.subplot(3, 3, 6)
    for (dataset_name, data), color in zip(all_data.items(), colors):
        signal_powers = []
        est_powers = []
        for batch_data in data:
            signal, estimates = _extract_signal_and_estimates(batch_data["batch_inputs"])
            sig_pwr = _compute_power(signal).numpy().flatten()
            est_pwr = _compute_power(estimates).numpy().flatten()
            signal_powers.extend(sig_pwr)
            est_powers.extend(est_pwr)
        ax6.scatter(signal_powers, est_powers, alpha=0.3, s=10, label=dataset_name, color=color)
    ax6.set_xlabel("Signal Power")
    ax6.set_ylabel("LS Estimates Power")
    ax6.set_title("Signal vs LS Estimates Power")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. SNR vs Signal Power
    ax7 = plt.subplot(3, 3, 7)
    for (dataset_name, data), color in zip(all_data.items(), colors):
        snr_vals = []
        sig_powers = []
        for batch_data in data:
            snr_batch = batch_data["batch_snr"].numpy()  # Shape: [B]
            signal, _ = _extract_signal_and_estimates(batch_data["batch_inputs"])
            sig_pwr = _compute_power(signal).numpy()  # Shape: [B, F, T]
            # Match dimensions: repeat SNR for each freq-time point
            for i, snr_val in enumerate(snr_batch):
                snr_vals.extend([snr_val] * sig_pwr[i].size)
                sig_powers.extend(sig_pwr[i].flatten())
        ax7.scatter(snr_vals, sig_powers, alpha=0.3, s=10, label=dataset_name, color=color)
    ax7.set_xlabel("SNR (dB)")
    ax7.set_ylabel("Signal Power")
    ax7.set_title("SNR vs Signal Power")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Power ratio (LS Estimates / Signal)
    ax8 = plt.subplot(3, 3, 8)
    for (dataset_name, data), color in zip(all_data.items(), colors):
        ratio_values = []
        for batch_data in data:
            signal, estimates = _extract_signal_and_estimates(batch_data["batch_inputs"])
            sig_pwr = _compute_power(signal).numpy().flatten()
            est_pwr = _compute_power(estimates).numpy().flatten()
            # Avoid division by zero
            ratio = np.divide(est_pwr, sig_pwr, where=sig_pwr > 1e-6, out=np.zeros_like(sig_pwr))
            ratio_values.extend(ratio)
        ax8.hist(ratio_values, bins=30, alpha=0.6, label=dataset_name, color=color)
    ax8.set_xlabel("Power Ratio (LS / Signal)")
    ax8.set_ylabel("Count")
    ax8.set_title("LS Estimates / Signal Power Ratio")
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Statistics Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis("tight")
    ax9.axis("off")

    table_data = []
    table_data.append(["Dataset", "SNR Mean", "SNR Std", "Signal Pwr", "Est Pwr"])
    for dataset_name in args.datasets:
        stats_list = [batch_data["stats"] for batch_data in all_data[dataset_name]]
        avg_stats = {key: np.mean([s[key] for s in stats_list]) for key in stats_list[0].keys()}
        table_data.append(
            [
                dataset_name,
                f"{avg_stats['snr_mean']:.2f}",
                f"{avg_stats['snr_std']:.2f}",
                f"{avg_stats['signal_power_mean']:.4f}",
                f"{avg_stats['estimates_power_mean']:.4f}",
            ]
        )

    table = ax9.table(cellText=table_data, cellLoc="center", loc="center", colWidths=[0.15, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    # Highlight header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"\n✓ Plots saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
