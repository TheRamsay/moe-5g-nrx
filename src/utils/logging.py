"""Logging utilities for experiment tracking."""

from typing import Any

import wandb
from omegaconf import DictConfig


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

    # Build run name from config
    run_name = f"{cfg.model.family}_{cfg.model.name}"
    if job_id:
        run_name = f"{run_name}_{job_id.split('.')[0]}"

    try:
        wandb.init(
            project=cfg.project.name,
            name=run_name,
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
