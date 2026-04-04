"""Logging utilities for experiment tracking."""

import os
from typing import Any

from omegaconf import DictConfig, OmegaConf

import wandb


def _cfg_get(cfg: DictConfig, path: str, default: Any = None) -> Any:
    """Safely read a nested config value."""

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


def setup_wandb(cfg: DictConfig, job_id: str | None = None, *, job_type: str = "train") -> bool:
    """Initialize wandb for experiment tracking.

    Args:
        cfg: Hydra configuration
        job_id: PBS job ID (if running on MetaCentrum)

    Returns:
        True if wandb is initialized successfully
    """
    if not bool(_cfg_get(cfg, "logging.use_wandb", False)):
        return False

    project_name = _cfg_get(cfg, "logging.project") or os.environ.get("WANDB_PROJECT") or _cfg_get(cfg, "project.name")
    entity = _cfg_get(cfg, "logging.entity") or os.environ.get("WANDB_ENTITY")
    log_code = bool(_cfg_get(cfg, "logging.log_code", True))

    # Get experiment metadata from config or env vars
    batch_name = _cfg_get(cfg, "experiment.batch_name") or os.environ.get("BATCH_NAME")

    exp_name = _cfg_get(cfg, "experiment.exp_name") or os.environ.get("EXP_NAME")

    # Build run name: prefer exp_name, fallback to model info
    if exp_name:
        run_name = exp_name
    else:
        model_family = str(cfg.model.family)
        model_name = str(cfg.model.name)
        if model_name.startswith(f"{model_family}_"):
            run_name = model_name
        else:
            run_name = f"{model_family}_{model_name}"

    if job_id:
        run_name = f"{run_name}_{job_id.split('.')[0]}"

    # Build tags from config
    tags: list[str] = [str(cfg.model.family)]
    if batch_name:
        tags.append(str(batch_name).replace("-", "_"))

    exp_tags = _cfg_get(cfg, "experiment.tags")
    if exp_tags:
        tags.extend([str(t) for t in exp_tags])

    logging_tags = _cfg_get(cfg, "logging.tags")
    if logging_tags:
        tags.extend([str(t) for t in logging_tags])

    tags = list(dict.fromkeys(tags))

    # Get notes
    notes = _cfg_get(cfg, "experiment.notes") or _cfg_get(cfg, "logging.notes")

    # Convert config to plain dict for wandb
    config_dict: dict[str, Any] = {}
    try:
        raw_config = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(raw_config, dict):
            for key, value in raw_config.items():
                config_dict[str(key)] = value
    except Exception:
        config_dict = {}

    try:
        wandb.init(
            project=str(project_name),
            entity=str(entity) if entity else None,
            name=run_name,
            group=batch_name,
            job_type=job_type,
            tags=tags,
            notes=notes,
            config=config_dict,
            save_code=log_code,
        )
        return True
    except Exception as e:
        print(f"[WARNING] Failed to initialize wandb: {e}")
        return False


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log metrics to wandb."""
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def finish_wandb() -> None:
    """Finish wandb run."""
    if wandb.run is not None:
        wandb.finish()
