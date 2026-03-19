"""Logging utilities for experiment tracking."""

import os
from typing import Any

import wandb
from omegaconf import DictConfig, OmegaConf


def setup_wandb(cfg: DictConfig, job_id: str | None = None) -> bool:
    """Initialize wandb for experiment tracking.

    Args:
        cfg: Hydra configuration
        job_id: PBS job ID (if running on MetaCentrum)

    Returns:
        True if wandb is initialized successfully
    """
    if not cfg.logging.use_wandb:
        return False

    # Get experiment metadata from config or env vars
    batch_name = cfg.experiment.get("batch_name") or os.environ.get("BATCH_NAME")
    exp_name = cfg.experiment.get("exp_name") or os.environ.get("EXP_NAME")

    # Build run name: prefer exp_name, fallback to model info
    if exp_name:
        run_name = exp_name
    else:
        run_name = f"{cfg.model.family}_{cfg.model.name}"

    if job_id:
        run_name = f"{run_name}_{job_id.split('.')[0]}"

    # Build tags from config
    tags = [cfg.model.family]
    if batch_name:
        tags.append(batch_name.replace("-", "_"))
    if cfg.experiment.get("tags"):
        tags.extend(cfg.experiment.tags)

    # Get notes
    notes = cfg.experiment.get("notes") or cfg.logging.get("notes")

    try:
        config_dict = dict(OmegaConf.to_container(cfg, resolve=True))
        wandb.init(
            project=cfg.project.name,
            name=run_name,
            group=batch_name,
            tags=tags,
            notes=notes,
            config=config_dict,
            save_code=True,
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
