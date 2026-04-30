#!/usr/bin/env python
"""Standalone evaluation script for the classical LS-MRC + max-log-LLR baseline.

Mirrors the metrics produced by scripts/evaluate.py (BLER, BER, channel MSE,
per-SNR-binned BLER) so the resulting JSON can be plotted alongside the neural
runs on the same Pareto curve.

Why a separate script: the LMMSE baseline has zero trainable parameters and
needs the per-sample SNR (the dataset stores it). Routing it through the
checkpoint-loading machinery in evaluate.py adds friction without value.

Usage:
    uv run python scripts/evaluate_lmmse.py \\
        --data-dir /storage/.../dense-v1/test \\
        --profiles uma tdlc \\
        --snr-bins 7 \\
        --batch-size 256 \\
        --out lmmse_eval_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import lmmse_forward  # noqa: E402
from src.data import CachedNRXBatch, build_cached_dataloader  # noqa: E402
from src.utils.metrics import compute_batch_error_metrics  # noqa: E402


@torch.no_grad()
def evaluate_lmmse_on_dataset(
    data_path: Path,
    device: torch.device,
    bits_per_symbol: int,
    num_rx_antennas: int,
    batch_size: int,
    snr_bins: int | None,
    mode: str = "ls_mrc",
) -> dict[str, Any]:
    """Run a classical baseline on one cached dataset and aggregate metrics.

    `mode` is forwarded to lmmse_forward — see baselines.lmmse for the options.
    """
    dataloader = build_cached_dataloader(
        path=data_path,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    channel_mse_fn = torch.nn.MSELoss()

    total_bce_loss = 0.0
    total_channel_mse = 0.0
    total_bit_errors = 0
    total_bits = 0
    total_block_errors = 0
    total_blocks = 0
    total_symbol_errors = 0
    total_symbols = 0
    num_batches = 0

    all_snr_db: list[float] = []
    all_bit_errors: list[int] = []
    all_bits_per_sample: list[int] = []
    all_block_errors: list[int] = []

    for batch in tqdm(dataloader, desc=f"LMMSE @ {data_path.name}", unit="batch"):
        batch: CachedNRXBatch
        features = batch.inputs.to(device)
        targets = batch.bit_labels.flatten(start_dim=1).to(device)
        channel_targets = batch.channel_target.to(device)
        snr_db = batch.snr_db.to(device)

        outputs = lmmse_forward(
            features,
            snr_db,
            num_rx_antennas=num_rx_antennas,
            mode=mode,
            channel_target=channel_targets if mode == "genie_mrc" else None,
        )
        logits = outputs["logits"]
        channel_estimate = outputs["channel_estimate"]

        bce_loss = bce_loss_fn(logits, targets)
        channel_mse = channel_mse_fn(channel_estimate, channel_targets)
        total_bce_loss += float(bce_loss.item())
        total_channel_mse += float(channel_mse.item())

        error_metrics = compute_batch_error_metrics(
            logits,
            targets,
            bits_per_symbol=bits_per_symbol,
            num_subcarriers=features.shape[2],
            num_ofdm_symbols=features.shape[3],
        )

        per_sample_bit_errors = error_metrics.bit_errors.sum(dim=1).cpu().numpy()
        per_sample_bits = targets.shape[1]
        per_sample_block_error = error_metrics.block_errors.cpu().numpy().astype(int)

        all_snr_db.extend(snr_db.cpu().numpy().tolist())
        all_bit_errors.extend(per_sample_bit_errors.tolist())
        all_bits_per_sample.extend([per_sample_bits] * snr_db.shape[0])
        all_block_errors.extend(per_sample_block_error.tolist())

        total_bit_errors += int(error_metrics.bit_errors.sum().item())
        total_bits += int(targets.numel())
        total_block_errors += int(error_metrics.block_errors.sum().item())
        total_blocks += int(error_metrics.block_errors.numel())
        total_symbol_errors += int(error_metrics.symbol_errors.sum().item())
        total_symbols += int(error_metrics.symbol_errors.numel())

        num_batches += 1

    results: dict[str, Any] = {
        "loss_bce": total_bce_loss / max(num_batches, 1),
        "channel_mse": total_channel_mse / max(num_batches, 1),
        "ber": total_bit_errors / max(total_bits, 1),
        "bler": total_block_errors / max(total_blocks, 1),
        "ser": total_symbol_errors / max(total_symbols, 1),
        "num_samples": total_blocks,
        "num_bit_errors": total_bit_errors,
        "num_block_errors": total_block_errors,
        # FLOPs: classical receiver has ~negligible compute vs neural; we don't
        # claim a precise number here, just record that it's effectively zero
        # compared to the neural baselines (~1.6 GFLOPs for dense_large).
        "realized_flops": 0.0,
    }

    if snr_bins is not None and snr_bins > 0:
        snr_array = np.array(all_snr_db)
        bit_errors_array = np.array(all_bit_errors)
        bits_per_sample_array = np.array(all_bits_per_sample)
        block_errors_array = np.array(all_block_errors)

        snr_min, snr_max = snr_array.min(), snr_array.max()
        bin_edges = np.linspace(snr_min, snr_max, snr_bins + 1)
        bin_indices = np.digitize(snr_array, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, snr_bins - 1)

        snr_binned_rows: list[dict[str, Any]] = []
        for i in range(snr_bins):
            mask = bin_indices == i
            n = int(mask.sum())
            if n == 0:
                continue
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            bin_ber = bit_errors_array[mask].sum() / bits_per_sample_array[mask].sum()
            bin_bler = block_errors_array[mask].mean()
            snr_binned_rows.append(
                {
                    "bin_index": i,
                    "snr_min": float(bin_edges[i]),
                    "snr_max": float(bin_edges[i + 1]),
                    "snr_center": float(bin_center),
                    "ber": float(bin_ber),
                    "bler": float(bin_bler),
                    "num_samples": n,
                }
            )
        results["snr_binned_rows"] = snr_binned_rows

    return results


def print_results(results: dict[str, Any], dataset_name: str) -> None:
    print(f"\n=== {dataset_name} ===")
    print(f"  BLER         : {results['bler']:.4f}")
    print(f"  BER          : {results['ber']:.4e}")
    print(f"  SER          : {results['ser']:.4f}")
    print(f"  Channel MSE  : {results['channel_mse']:.4e}")
    print(f"  num samples  : {results['num_samples']}")
    rows = results.get("snr_binned_rows", [])
    if rows:
        print("  Per-SNR BLER:")
        for row in rows:
            print(
                f"    [{row['snr_min']:6.2f}, {row['snr_max']:6.2f}] dB "
                f"(n={row['num_samples']:5d}): BLER={row['bler']:.4f}, BER={row['ber']:.4e}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Directory containing {profile}.pt cached test datasets"
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["uma", "tdlc"],
        help="Which channel profiles to evaluate (must match {profile}.pt files)",
    )
    parser.add_argument("--snr-bins", type=int, default=7, help="Number of SNR bins for per-SNR breakdown")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-rx-antennas", type=int, default=4)
    parser.add_argument("--bits-per-symbol", type=int, default=4)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--out", type=Path, default=Path("lmmse_eval_results.json"))
    parser.add_argument(
        "--mode",
        default="ls_mrc",
        choices=["ls_mrc", "genie_mrc", "single_ant"],
        help="Which classical baseline to run.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available; falling back to CPU", file=sys.stderr)
        device = torch.device("cpu")

    all_results: dict[str, dict[str, Any]] = {}
    for profile in args.profiles:
        data_path = args.data_dir / f"{profile}.pt"
        if not data_path.exists():
            print(f"[WARNING] missing dataset {data_path}, skipping", file=sys.stderr)
            continue
        results = evaluate_lmmse_on_dataset(
            data_path=data_path,
            device=device,
            bits_per_symbol=args.bits_per_symbol,
            num_rx_antennas=args.num_rx_antennas,
            batch_size=args.batch_size,
            snr_bins=args.snr_bins,
            mode=args.mode,
        )
        all_results[profile] = results
        print_results(results, dataset_name=profile)

    if all_results:
        avg_bler = float(np.mean([r["bler"] for r in all_results.values()]))
        avg_ber = float(np.mean([r["ber"] for r in all_results.values()]))
        print("\n=== Summary across profiles ===")
        print(f"  Avg BLER : {avg_bler:.4f}")
        print(f"  Avg BER  : {avg_ber:.4e}")
        all_results["_summary"] = {"avg_bler": avg_bler, "avg_ber": avg_ber}

    args.out.write_text(json.dumps(all_results, indent=2))
    print(f"\n[INFO] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
