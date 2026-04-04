from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

import wandb


def _cfg_get(cfg: DictConfig, path: str, default: Any = None) -> Any:
    current: Any = cfg
    for key in path.split("."):
        if current is None:
            return default
        if isinstance(current, DictConfig):
            if key not in current:
                return default
            current = current[key]
            continue
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
            continue
        return default
    return current


def _sanitize_artifact_component(value: str) -> str:
    sanitized = []
    for char in value.lower():
        if char.isalnum() or char in {"-", "_", "."}:
            sanitized.append(char)
        else:
            sanitized.append("-")
    return "".join(sanitized).strip("-") or "artifact"


def build_dataset_artifact_name(split: str, profile: str) -> str:
    return f"dataset-{_sanitize_artifact_component(split)}-{_sanitize_artifact_component(profile)}"


def _artifact_ref(project: str, entity: str | None, name: str, alias: str) -> str:
    if entity:
        return f"{entity}/{project}/{name}:{alias}"
    return f"{project}/{name}:{alias}"


def setup_generation_wandb(cfg: DictConfig) -> bool:
    if not bool(_cfg_get(cfg, "logging.use_wandb", False)) or not bool(_cfg_get(cfg, "generation.log_to_wandb", False)):
        return False

    project = str(_cfg_get(cfg, "logging.project") or _cfg_get(cfg, "project.name", "moe-5g-nrx"))
    entity = _cfg_get(cfg, "logging.entity")
    split = str(_cfg_get(cfg, "generation.split", "both"))
    profiles = list(_cfg_get(cfg, "generation.profiles", []))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"datasets_{split}_{'-'.join(str(profile) for profile in profiles)}_{stamp}"

    raw_config = OmegaConf.to_container(cfg, resolve=True)
    config_payload = raw_config if isinstance(raw_config, dict) else {}

    wandb.init(
        project=project,
        entity=str(entity) if entity else None,
        name=run_name,
        group="dataset-generation",
        tags=["data", "generation"],
        job_type="data_generation",
        config=config_payload,
    )
    return True


def log_dataset_artifact(
    dataset_path: str | Path,
    *,
    split: str,
    profile: str,
    metadata: dict[str, Any],
    aliases: list[str] | None = None,
) -> str | None:
    if wandb.run is None:
        return None

    path = Path(dataset_path)
    artifact_name = build_dataset_artifact_name(split, profile)
    artifact = wandb.Artifact(
        artifact_name,
        type="dataset",
        description=f"Cached {split} dataset for {profile}",
        metadata={
            "split": split,
            "profile": profile,
            "path": str(path),
            **metadata,
        },
    )
    artifact.add_file(str(path), name=path.name)
    alias_list = aliases or ["latest"]
    wandb.log_artifact(artifact, aliases=alias_list)
    ref = _artifact_ref(wandb.run.project, wandb.run.entity, artifact_name, alias_list[0])
    wandb.run.summary[f"artifacts/{split}/{profile}"] = ref
    return ref


def build_checkpoint_artifact_name(run_id: str) -> str:
    return f"model-{_sanitize_artifact_component(run_id)}"


def build_checkpoint_artifact_ref(project: str, entity: str | None, run_id: str, alias: str = "latest") -> str:
    return _artifact_ref(project, entity, build_checkpoint_artifact_name(run_id), alias)


def log_checkpoint_artifact(
    checkpoint_path: str | Path,
    *,
    metadata: dict[str, Any],
    aliases: list[str] | None = None,
) -> str | None:
    if wandb.run is None:
        return None

    path = Path(checkpoint_path)
    artifact_name = build_checkpoint_artifact_name(wandb.run.id)
    artifact = wandb.Artifact(
        artifact_name,
        type="model",
        description=f"Checkpoint for run {wandb.run.name}",
        metadata={"path": str(path), **metadata},
    )
    artifact.add_file(str(path), name=path.name)
    alias_list = aliases or ["latest", "final"]
    wandb.log_artifact(artifact, aliases=alias_list)
    ref = _artifact_ref(wandb.run.project, wandb.run.entity, artifact_name, alias_list[0])
    wandb.run.summary["artifacts/checkpoint"] = ref
    return ref


def use_artifact_path(artifact_ref: str, *, expected_filename: str | None = None) -> Path:
    if wandb.run is None:
        raise RuntimeError("wandb.run is required to use artifacts")

    artifact = wandb.run.use_artifact(artifact_ref)
    download_dir = Path(artifact.download())
    if expected_filename is not None:
        candidate = download_dir / expected_filename
        if candidate.exists():
            return candidate

    files = [path for path in download_dir.iterdir() if path.is_file()]
    if len(files) == 1:
        return files[0]
    raise FileNotFoundError(f"Unable to resolve downloaded artifact file in {download_dir}")
