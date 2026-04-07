#!/usr/bin/env python


from __future__ import annotations

import gc
import json
import multiprocessing as mp
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra import compose
from omegaconf import DictConfig

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.wandb_artifacts import log_dataset_artifact, setup_generation_wandb  # noqa: E402


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


# ---------------------------------------------------------------------------
# Subprocess-isolated chunk generation
# Each chunk runs in a fresh Python process (spawn) so TF memory is fully
# released by the OS when the child exits. This prevents the TF allocator
# from accumulating RAM across chunks, which otherwise stalls generation.
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 100_000
_CONFIG_DIR = str(PROJECT_ROOT / "conf")


def _chunk_worker(args_json: str) -> None:
    """Entry point for a spawned subprocess that generates one chunk.

    Fully self-contained: initializes Hydra, builds dataloader, generates
    samples, and saves them to a temp shard file. Called via multiprocessing
    with the 'spawn' start method so TF starts fresh in every child.
    """
    args = json.loads(args_json)
    profile: str = args["profile"]
    chunk_size: int = args["chunk_size"]
    batch_size: int = args["batch_size"]
    seed: int = args["seed"]
    output_path: str = args["output_path"]
    config_dir: str = args["config_dir"]
    project_root: str = args["project_root"]
    src_root: str = args["src_root"]

    # Re-add project paths (subprocess doesn't inherit sys.path changes).
    for p in [src_root, project_root]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Force CPU-only TF BEFORE any TF/Sionna import so the GPU device is
    # never claimed. This must happen before the first 'import tensorflow'.
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    from hydra import initialize_config_dir
    from omegaconf import open_dict

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        from hydra import compose as hcompose

        num_batches = (chunk_size + batch_size - 1) // batch_size
        cfg = hcompose(
            config_name="config",
            overrides=[
                f"dataset={profile}",
                f"training.batch_size={batch_size}",
                f"runtime.seed={seed}",
                "dataset.diagnostics.sample_preview=false",
            ],
        )
        with open_dict(cfg):
            cfg.data.num_subcarriers = int(cfg.dataset.num_subcarriers)
            cfg.data.num_ofdm_symbols = int(cfg.dataset.num_ofdm_symbols)
            cfg.dataset.max_batches_per_epoch = num_batches

        from src.data import build_dataloader

        dataloader = build_dataloader(cfg)

        # Pre-allocate to avoid 2× peak RAM from torch.cat.
        inputs: torch.Tensor | None = None
        bit_labels: torch.Tensor | None = None
        channel_target: torch.Tensor | None = None
        snr_db: torch.Tensor | None = None
        collected = 0

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            remaining = chunk_size - collected
            take = min(batch_size, remaining)

            b_inp = batch.inputs[:take].cpu()
            b_bit = batch.bit_labels[:take].cpu()
            b_ch = batch.channel_target[:take].cpu()
            b_snr = batch.snr_db[:take].cpu()

            if inputs is None:
                inputs = torch.empty(chunk_size, *b_inp.shape[1:], dtype=b_inp.dtype)
                bit_labels = torch.empty(chunk_size, *b_bit.shape[1:], dtype=b_bit.dtype)
                channel_target = torch.empty(chunk_size, *b_ch.shape[1:], dtype=b_ch.dtype)
                snr_db = torch.empty(chunk_size, *b_snr.shape[1:], dtype=b_snr.dtype)

            inputs[collected : collected + take] = b_inp
            bit_labels[collected : collected + take] = b_bit
            channel_target[collected : collected + take] = b_ch
            snr_db[collected : collected + take] = b_snr

            collected += take
            if collected >= chunk_size:
                break

    assert inputs is not None
    torch.save(
        {
            "inputs": inputs,
            "bit_labels": bit_labels,
            "channel_target": channel_target,
            "snr_db": snr_db,
        },
        output_path,
    )


def generate_dataset(
    profile: str,
    num_samples: int,
    batch_size: int,
    seed: int,
    desc: str = "",
) -> dict[str, Any]:
    """Generate a dataset for a single channel profile.

    Internally splits generation into _CHUNK_SIZE-sample chunks, each run in
    an isolated subprocess (spawn). This ensures TF/Sionna memory is fully
    reclaimed by the OS after each chunk, preventing the accumulation that
    stalls generation on large datasets.

    Args:
        profile: Channel profile name (awgn, uma, tdlc).
        num_samples: Total number of samples to generate.
        batch_size: Batch size for generation.
        seed: Random seed for reproducibility.
        desc: Description for progress bar.

    Returns:
        Dictionary with tensors and metadata ready for saving.
    """
    chunks_needed = (num_samples + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    progress_desc = desc or f"Generating {profile}"
    ctx = mp.get_context("spawn")

    # Pre-allocate output tensors; shapes determined from first shard.
    inputs: torch.Tensor | None = None
    bit_labels: torch.Tensor | None = None
    channel_target: torch.Tensor | None = None
    snr_db: torch.Tensor | None = None

    total_collected = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        for chunk_idx in range(chunks_needed):
            chunk_size = min(_CHUNK_SIZE, num_samples - total_collected)
            chunk_seed = seed + chunk_idx
            shard_path = str(Path(tmp_dir) / f"shard_{chunk_idx:04d}.pt")
            chunk_label = f" [chunk {chunk_idx + 1}/{chunks_needed}]" if chunks_needed > 1 else ""
            print(f"  {progress_desc}{chunk_label}: generating {chunk_size} samples (seed={chunk_seed})", flush=True)

            args_json = json.dumps(
                {
                    "profile": profile,
                    "chunk_size": chunk_size,
                    "batch_size": batch_size,
                    "seed": chunk_seed,
                    "output_path": shard_path,
                    "config_dir": _CONFIG_DIR,
                    "project_root": str(PROJECT_ROOT),
                    "src_root": str(SRC_ROOT),
                }
            )

            proc = ctx.Process(target=_chunk_worker, args=(args_json,))
            proc.start()
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"Chunk {chunk_idx} worker exited with code {proc.exitcode}")

            shard: dict[str, torch.Tensor] = torch.load(shard_path)
            n = shard["inputs"].shape[0]

            if inputs is None:
                inputs = torch.empty(num_samples, *shard["inputs"].shape[1:], dtype=shard["inputs"].dtype)
                bit_labels = torch.empty(num_samples, *shard["bit_labels"].shape[1:], dtype=shard["bit_labels"].dtype)
                channel_target = torch.empty(
                    num_samples, *shard["channel_target"].shape[1:], dtype=shard["channel_target"].dtype
                )
                snr_db = torch.empty(num_samples, *shard["snr_db"].shape[1:], dtype=shard["snr_db"].dtype)

            inputs[total_collected : total_collected + n] = shard["inputs"]
            bit_labels[total_collected : total_collected + n] = shard["bit_labels"]
            channel_target[total_collected : total_collected + n] = shard["channel_target"]
            snr_db[total_collected : total_collected + n] = shard["snr_db"]

            del shard
            gc.collect()
            total_collected += n

    assert inputs is not None, "No batches were generated"

    # Build metadata via the parent's already-initialized Hydra context.
    cfg = build_config(profile, batch_size, seed)
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


def _log_dataset_to_wandb(
    cfg: DictConfig, split_name: str, profile: str, output_path: Path, data: dict[str, Any]
) -> None:
    if wandb.run is None:
        return

    artifact_aliases = [str(alias) for alias in cfg.generation.get("artifact_aliases", ["latest"])]
    artifact_ref = log_dataset_artifact(
        output_path,
        split=split_name,
        profile=profile,
        metadata=data["metadata"],
        aliases=artifact_aliases,
    )
    if artifact_ref is not None:
        print(f"  Logged WandB artifact: {artifact_ref}")


def _resolve_split(split: str) -> list[tuple[str, int]]:
    from src.data.constants import TEST_SEED_OFFSET, TRAIN_DATASET_SEED_OFFSET, VAL_SEED_OFFSET

    split_normalized = split.lower()
    if split_normalized == "train":
        return [("train", TRAIN_DATASET_SEED_OFFSET)]
    if split_normalized == "val":
        return [("val", VAL_SEED_OFFSET)]
    if split_normalized == "test":
        return [("test", TEST_SEED_OFFSET)]
    if split_normalized == "both":
        return [("val", VAL_SEED_OFFSET), ("test", TEST_SEED_OFFSET)]
    raise ValueError(f"Invalid split '{split}'. Expected one of: train, val, test, both")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    generation_cfg = cfg.generation
    use_wandb = setup_generation_wandb(cfg)

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
            output_path = split_dir / f"{profile}.pt"
            save_dataset(data, output_path)
            _log_dataset_to_wandb(cfg, split_name, profile, output_path, data)
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
            output_path = split_dir / "mixed.pt"
            save_dataset(data, output_path)
            _log_dataset_to_wandb(cfg, split_name, "mixed", output_path, data)
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

    if use_wandb and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
