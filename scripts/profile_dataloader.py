#!/usr/bin/env python
"""Profile Sionna data generation speed vs GPU training step speed.

Usage:
    uv run python scripts/profile_dataloader.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.data import build_dataloader  # noqa: E402
from src.models import build_model_from_config  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_WARMUP = 5
N_PROFILE = 20
BATCH_SIZE = 64


def _make_sionna_cfg(profile: str = "tdlc"):
    """Minimal OmegaConf-like config for the mixed dataloader."""
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "data": {
                "input_channels": 16,
                "num_rx_antennas": 4,
                "num_subcarriers": 128,
                "num_ofdm_symbols": 14,
                "bits_per_symbol": 4,
            },
            "dataset": {
                "name": profile,
                "channel_profile": profile,
                "num_subcarriers": 128,
                "num_ofdm_symbols": 14,
                "modulation": "16qam",
                "snr_db": {"min": -10.0, "max": 20.0},
                "delay_spread_s": 5.0e-7,
                "min_speed_mps": 0.0,
                "max_speed_mps": 25.0,
                "num_sinrs": 1,
                "estimation_noise_std": 0.1,
                "pilot_pattern": "kronecker",
                "pilot_ofdm_symbol_indices": [2, 11],
                "fft_size": 128,
                "subcarrier_spacing_hz": 30000.0,
                "cyclic_prefix_length": 20,
                "num_guard_carriers": [5, 6],
                "dc_null": True,
                "use_pilots": True,
                "normalize_channel": True,
                "max_batches_per_epoch": 0,
                "loader": {
                    "num_workers": 0,
                    "pin_memory": False,
                    "prefetch_factor": 2,
                    "persistent_workers": True,
                },
            },
            "training": {
                "batch_size": BATCH_SIZE,
                "overfit_single_batch": False,
            },
            "runtime": {"seed": 67},
        }
    )


def _make_minimal_model():
    config = {
        "model": {
            "name": "static_dense_nrx",
            "family": "static_dense",
            "backbone": {
                "stem_hidden_dims": [64, 64],
                "state_dim": 56,
                "num_cnn_blocks": 8,
                "block_hidden_dim": 64,
                "readout_hidden_dim": 128,
            },
        },
        "data": {
            "input_channels": 16,
            "num_rx_antennas": 4,
            "num_subcarriers": 128,
            "num_ofdm_symbols": 14,
            "bits_per_symbol": 4,
        },
    }
    from omegaconf import OmegaConf

    return build_model_from_config(OmegaConf.create(config))


def profile_sionna(num_workers: int = 0):
    print(f"\n{'='*55}")
    print(f"  Sionna generation  (num_workers={num_workers})")
    print(f"{'='*55}")

    cfg = _make_sionna_cfg()
    cfg.dataset.loader.num_workers = num_workers

    loader = build_dataloader(cfg)

    # Warmup
    print(f"  Warming up ({N_WARMUP} batches)...")
    for i, _ in enumerate(loader):
        if i >= N_WARMUP - 1:
            break

    # Profile
    print(f"  Profiling ({N_PROFILE} batches)...")
    times = []
    t0 = time.perf_counter()
    for i, _ in enumerate(loader):
        t1 = time.perf_counter()
        times.append(t1 - t0)
        t0 = t1
        if i >= N_PROFILE - 1:
            break

    avg = sum(times) / len(times)
    print(f"  avg batch time : {avg*1000:.1f} ms  ({1/avg:.1f} batches/s)")
    print(f"  samples/sec    : {BATCH_SIZE/avg:.0f}")
    return avg


def profile_training_step(model: torch.nn.Module):
    print(f"\n{'='*55}")
    print(f"  GPU training step  (large model, batch={BATCH_SIZE})")
    print(f"{'='*55}")

    model.to(DEVICE).train()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Make a fake batch matching the expected input shape
    # inputs: [B, input_channels, num_subcarriers, num_ofdm_symbols]
    x = torch.randn(BATCH_SIZE, 16, 128, 14, device=DEVICE)
    targets = torch.randint(0, 2, (BATCH_SIZE, 4 * 128 * 14), device=DEVICE).float()

    # Warmup
    print(f"  Warming up ({N_WARMUP} steps)...")
    for _ in range(N_WARMUP):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out["logits"], targets)
        loss.backward()
        optimizer.step()

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    # Profile
    print(f"  Profiling ({N_PROFILE} steps)...")
    times = []
    for _ in range(N_PROFILE):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out["logits"], targets)
        loss.backward()
        optimizer.step()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    print(f"  avg step time  : {avg*1000:.1f} ms  ({1/avg:.1f} steps/s)")
    return avg


def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Profile training step first (doesn't need Sionna)
    model = _make_minimal_model()
    step_time = profile_training_step(model)
    del model

    # Profile Sionna with num_workers=0 (current default)
    gen_time_0 = profile_sionna(num_workers=0)

    # Profile Sionna with num_workers=2
    gen_time_2 = profile_sionna(num_workers=2)

    # Summary
    print(f"\n{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}")
    print(f"  GPU training step        : {step_time*1000:7.1f} ms")
    print(f"  Sionna gen (workers=0)   : {gen_time_0*1000:7.1f} ms")
    print(f"  Sionna gen (workers=2)   : {gen_time_2*1000:7.1f} ms")
    ratio_0 = gen_time_0 / step_time
    ratio_2 = gen_time_2 / step_time
    print(f"\n  Gen/step ratio (w=0)     : {ratio_0:.1f}x  ", end="")
    if ratio_0 < 1.5:
        print("✓ gen is fast, workers should help")
    elif ratio_0 < 5.0:
        print("⚠ gen is slower, some benefit from workers")
    else:
        print("✗ gen is bottleneck, workers won't fully solve it")
    print(f"  Gen/step ratio (w=2)     : {ratio_2:.1f}x")
    print()


if __name__ == "__main__":
    main()
