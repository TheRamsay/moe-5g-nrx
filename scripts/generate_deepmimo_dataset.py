#!/usr/bin/env python


from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, open_dict

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.constants import TEST_SEED_OFFSET, TRAIN_DATASET_SEED_OFFSET, VAL_SEED_OFFSET  # noqa: E402
from src.data.deepmimo_generator import DeepMIMOGenerationConfig, generate_deepmimo_arrow_dataset  # noqa: E402
from src.utils.wandb_artifacts import log_dataset_artifact, setup_generation_wandb  # noqa: E402


def _resolve_split(split: str) -> list[tuple[str, int]]:
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


def _log_dataset_to_wandb(
    cfg: DictConfig,
    split_name: str,
    profile: str,
    output_path: Path,
    metadata: dict[str, object],
) -> None:
    if wandb.run is None:
        return

    artifact_aliases = [str(alias) for alias in cfg.generation.get("artifact_aliases", ["latest"])]
    artifact_ref = log_dataset_artifact(
        output_path,
        split=split_name,
        profile=profile,
        metadata=metadata,
        aliases=artifact_aliases,
    )
    if artifact_ref is not None:
        print(f"  Logged WandB artifact: {artifact_ref}")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    generation_cfg = cfg.generation
    deepmimo_cfg = generation_cfg.get("deepmimo", {})
    profile_name = str(deepmimo_cfg.get("scenario"))
    if not profile_name or profile_name.strip() == "":
        raise ValueError("DeepMIMO scenario name must be provided in generation.deepmimo.scenario")
    scenario = deepmimo_cfg.get("scenario")
    if scenario is None or str(scenario).strip() == "":
        raise ValueError("generation.deepmimo.scenario must be set (scenario name or absolute path).")

    with open_dict(cfg):
        cfg.generation.profiles = [profile_name]
        cfg.generation.include_mixed = False

    use_wandb = setup_generation_wandb(cfg)

    output_dir = Path(str(generation_cfg.output_dir))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    num_samples = int(generation_cfg.num_samples)
    if num_samples <= 0:
        raise ValueError("generation.num_samples must be > 0")

    split = str(generation_cfg.split)
    base_seed = int(generation_cfg.base_seed)
    splits_to_generate = _resolve_split(split)

    print("=" * 60)
    print("DeepMIMO OOD Dataset Generation")
    print("=" * 60)
    print(f"  Scenario: {scenario}")
    print(f"  Profile name: {profile_name}")
    print(f"  Output directory: {output_dir}")
    print(f"  Samples per split: {num_samples}")
    print(f"  Splits: {[split_name for split_name, _ in splits_to_generate]}")
    print(f"  Base seed: {base_seed}")
    print("=" * 60)

    for split_name, split_seed_offset in splits_to_generate:
        split_seed = base_seed + split_seed_offset
        split_dir = output_dir / split_name
        output_path = split_dir / profile_name
        print(f"\n[{split_name}/{profile_name}] Generating Arrow dataset at {output_path} (seed={split_seed})")

        user_rows_cfg = deepmimo_cfg.get("user_rows")
        user_rows = None if user_rows_cfg is None else tuple(int(x) for x in user_rows_cfg)

        generator_cfg = DeepMIMOGenerationConfig(
            scenario=str(scenario),
            num_samples=num_samples,
            num_rx_antennas=int(cfg.data.num_rx_antennas),
            num_subcarriers=int(cfg.data.num_subcarriers),
            num_ofdm_symbols=int(cfg.data.num_ofdm_symbols),
            bits_per_symbol=int(cfg.data.bits_per_symbol),
            snr_db_min=float(deepmimo_cfg.get("snr_db_min", cfg.dataset.snr_db.min)),
            snr_db_max=float(deepmimo_cfg.get("snr_db_max", cfg.dataset.snr_db.max)),
            estimation_noise_std=float(deepmimo_cfg.get("estimation_noise_std", cfg.dataset.estimation_noise_std)),
            pilot_spacing_freq=int(cfg.model.backbone.pilot_subcarrier_spacing),
            pilot_spacing_time=tuple(int(x) for x in cfg.model.backbone.pilot_symbol_indices),
            seed=split_seed,
            profile_name=profile_name,
            modulation=str(cfg.dataset.modulation),
            active_users_only=bool(deepmimo_cfg.get("active_users_only", True)),
            dataset_folder=str(deepmimo_cfg.get("dataset_folder", "./data/deepmimov3/")),
            active_bs=tuple(int(x) for x in deepmimo_cfg.get("active_bs", [1])),
            user_rows=user_rows,
            user_subsampling=float(deepmimo_cfg.get("user_subsampling", 1.0)),
            num_paths=int(deepmimo_cfg.get("num_paths", 10)),
            use_all_rows_if_unspecified=bool(deepmimo_cfg.get("use_all_rows_if_unspecified", True)),
        )
        metadata = generate_deepmimo_arrow_dataset(generator_cfg, output_path)
        print(f"  Saved Arrow dataset: {output_path} ({metadata['num_samples']} samples)")
        _log_dataset_to_wandb(cfg, split_name, profile_name, output_path, metadata)

    print(f"\nDeepMIMO generation complete. Output root: {output_dir}")
    if use_wandb and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
