#!/usr/bin/env python


from __future__ import annotations

import gc
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra import compose
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_config(profile: str, batch_size: int, seed: int) -> DictConfig:
    """Build Hydra config for a specific channel profile."""
    cfg = compose(
        config_name="config",
        overrides=[
            f"dataset={profile}",
            f"training.batch_size={batch_size}",
            f"runtime.seed={seed}",
            "dataset.diagnostics.sample_preview=false",
        ],
    )
    return cfg


def generate_dataset(
    profile: str,
    num_samples: int,
    batch_size: int,
    seed: int,
    desc: str = "",
) -> dict[str, Any]:
    """Generate a dataset for a single channel profile.

    Args:
        profile: Channel profile name (awgn, uma, tdlc).
        num_samples: Total number of samples to generate.
        batch_size: Batch size for generation.
        seed: Random seed for reproducibility.
        desc: Description for progress bar.

    Returns:
        Dictionary with tensors and metadata ready for saving.
    """
    from src.data import build_dataloader

    num_batches = (num_samples + batch_size - 1) // batch_size
    cfg = build_config(profile, batch_size, seed)

    # Update config to sync dimensions
    with open_dict(cfg):
        cfg.data.num_subcarriers = int(cfg.dataset.num_subcarriers)
        cfg.data.num_ofdm_symbols = int(cfg.dataset.num_ofdm_symbols)
        # Dataloader uses this as a hard stop; raise it to satisfy requested samples.
        cfg.dataset.max_batches_per_epoch = int(num_batches)

    dataloader = build_dataloader(cfg)

    inputs_list: list[torch.Tensor] = []
    bit_labels_list: list[torch.Tensor] = []
    channel_target_list: list[torch.Tensor] = []
    snr_db_list: list[torch.Tensor] = []

    collected = 0

    progress_desc = desc or f"Generating {profile}"
    with tqdm(total=num_samples, desc=progress_desc, unit="samples") as pbar:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            # Determine how many samples to take from this batch
            remaining = num_samples - collected
            take = min(batch_size, remaining)

            inputs_list.append(batch.inputs[:take])
            bit_labels_list.append(batch.bit_labels[:take])
            channel_target_list.append(batch.channel_target[:take])
            snr_db_list.append(batch.snr_db[:take])

            collected += take
            pbar.update(take)

            if collected >= num_samples:
                break

    # Concatenate all batches
    inputs = torch.cat(inputs_list, dim=0)
    bit_labels = torch.cat(bit_labels_list, dim=0)
    channel_target = torch.cat(channel_target_list, dim=0)
    snr_db = torch.cat(snr_db_list, dim=0)

    # Build metadata
    simulator_config = {
        "num_subcarriers": int(cfg.dataset.num_subcarriers),
        "num_ofdm_symbols": int(cfg.dataset.num_ofdm_symbols),
        "snr_db_min": float(cfg.dataset.snr_db.min),
        "snr_db_max": float(cfg.dataset.snr_db.max),
        "modulation": str(cfg.dataset.modulation),
        "channel_profile": str(cfg.dataset.channel_profile),
    }

    metadata = {
        "channel_profile": profile,
        "modulation": str(cfg.dataset.modulation),
        "num_samples": int(inputs.shape[0]),
        "seed": seed,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "simulator_config": simulator_config,
    }

    return {
        "inputs": inputs,
        "bit_labels": bit_labels,
        "channel_target": channel_target,
        "snr_db": snr_db,
        "metadata": metadata,
    }


def generate_mixed_dataset(
    profiles: list[str],
    num_samples: int,
    batch_size: int,
    seed: int,
    desc: str = "",
) -> dict[str, Any]:
    """Generate a mixed dataset sampling equally from all profiles.

    Args:
        profiles: List of channel profiles to include.
        num_samples: Total number of samples in the mixed dataset.
        batch_size: Batch size for generation.
        seed: Base random seed.
        desc: Description for progress bar.

    Returns:
        Dictionary with tensors and metadata ready for saving.
    """
    samples_per_profile = num_samples // len(profiles)
    remainder = num_samples % len(profiles)

    all_inputs: list[torch.Tensor] = []
    all_bit_labels: list[torch.Tensor] = []
    all_channel_target: list[torch.Tensor] = []
    all_snr_db: list[torch.Tensor] = []
    profile_configs: dict[str, dict[str, Any]] = {}

    for i, profile in enumerate(profiles):
        # Distribute remainder samples across first profiles
        profile_samples = samples_per_profile + (1 if i < remainder else 0)
        profile_seed = seed + i * 100_000

        progress_desc = f"{desc} [{profile}]" if desc else f"Mixed [{profile}]"
        data = generate_dataset(
            profile=profile,
            num_samples=profile_samples,
            batch_size=batch_size,
            seed=profile_seed,
            desc=progress_desc,
        )

        all_inputs.append(data["inputs"])
        all_bit_labels.append(data["bit_labels"])
        all_channel_target.append(data["channel_target"])
        all_snr_db.append(data["snr_db"])
        profile_configs[profile] = data["metadata"]["simulator_config"]

    # Concatenate and shuffle
    inputs = torch.cat(all_inputs, dim=0)
    bit_labels = torch.cat(all_bit_labels, dim=0)
    channel_target = torch.cat(all_channel_target, dim=0)
    snr_db = torch.cat(all_snr_db, dim=0)

    # Shuffle with fixed seed for reproducibility
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(inputs))
    perm_tensor = torch.from_numpy(perm)

    inputs = inputs[perm_tensor]
    bit_labels = bit_labels[perm_tensor]
    channel_target = channel_target[perm_tensor]
    snr_db = snr_db[perm_tensor]

    metadata = {
        "channel_profile": "mixed",
        "modulation": "qam16",
        "num_samples": int(inputs.shape[0]),
        "seed": seed,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "profiles_included": profiles,
        "samples_per_profile": samples_per_profile,
        "profile_configs": profile_configs,
    }

    return {
        "inputs": inputs,
        "bit_labels": bit_labels,
        "channel_target": channel_target,
        "snr_db": snr_db,
        "metadata": metadata,
    }


def save_dataset(data: dict[str, Any], output_path: Path) -> None:
    """Save dataset to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB, {data['metadata']['num_samples']} samples)")


def _resolve_split(split: str) -> list[tuple[str, int]]:
    from src.data.constants import TEST_SEED_OFFSET, VAL_SEED_OFFSET

    split_normalized = split.lower()
    if split_normalized == "val":
        return [("val", VAL_SEED_OFFSET)]
    if split_normalized == "test":
        return [("test", TEST_SEED_OFFSET)]
    if split_normalized == "both":
        return [("val", VAL_SEED_OFFSET), ("test", TEST_SEED_OFFSET)]
    raise ValueError(f"Invalid split '{split}'. Expected one of: val, test, both")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    generation_cfg = cfg.generation

    output_dir = Path(str(generation_cfg.output_dir))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    num_samples = int(generation_cfg.num_samples)
    batch_size = int(generation_cfg.batch_size)
    split = str(generation_cfg.split)
    profiles = list(generation_cfg.profiles)
    include_mixed = bool(generation_cfg.include_mixed)
    base_seed = int(generation_cfg.base_seed)

    if num_samples <= 0:
        raise ValueError("generation.num_samples must be > 0")
    if batch_size <= 0:
        raise ValueError("generation.batch_size must be > 0")

    splits_to_generate = _resolve_split(split)

    print("=" * 60)
    print("Dataset Generation Configuration")
    print("=" * 60)
    print(f"  Output directory: {output_dir}")
    print(f"  Samples per dataset: {num_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Splits: {[s[0] for s in splits_to_generate]}")
    print(f"  Profiles: {profiles}")
    print(f"  Include mixed: {include_mixed}")
    print(f"  Base seed: {base_seed}")
    print("=" * 60)

    for split_name, seed_offset in splits_to_generate:
        split_dir = output_dir / split_name
        split_seed = base_seed + seed_offset

        print(f"\n{'=' * 60}")
        print(f"Generating {split_name.upper()} datasets (seed offset: {seed_offset})")
        print(f"{'=' * 60}")

        # Generate per-profile datasets
        sorted_profiles = sorted(profiles)
        for profile in profiles:
            profile_idx = sorted_profiles.index(profile)
            profile_seed = split_seed + profile_idx * 10_000
            print(f"\n[{split_name}/{profile}] Seed: {profile_seed}")

            data = generate_dataset(
                profile=profile,
                num_samples=num_samples,
                batch_size=batch_size,
                seed=profile_seed,
                desc=f"{split_name}/{profile}",
            )
            save_dataset(data, split_dir / f"{profile}.pt")
            del data
            gc.collect()  # Free memory after each profile

        # Generate mixed dataset
        if include_mixed and len(profiles) > 1:
            mixed_seed = split_seed + 90_000
            print(f"\n[{split_name}/mixed] Seed: {mixed_seed}")

            data = generate_mixed_dataset(
                profiles=profiles,
                num_samples=num_samples,
                batch_size=batch_size,
                seed=mixed_seed,
                desc=f"{split_name}/mixed",
            )
            save_dataset(data, split_dir / "mixed.pt")
            del data
            gc.collect()

    print(f"\n{'=' * 60}")
    print("Generation complete!")
    print(f"{'=' * 60}")
    print(f"\nDatasets saved to: {output_dir}")
    print("\nTo use in training, update config.yaml:")
    suggested_name = "mixed.pt" if include_mixed and len(profiles) > 1 else f"{profiles[0]}.pt"
    generated_splits = {name for name, _ in splits_to_generate}
    if "val" in generated_splits:
        print(f"  validation.dataset_path: {output_dir}/val/{suggested_name}")
    if "test" in generated_splits:
        print(f"  evaluation.dataset_path: {output_dir}/test/{suggested_name}")


if __name__ == "__main__":
    main()
