"""Logging utilities for experiment tracking."""

import os
from pathlib import Path
from typing import Any

import wandb
from omegaconf import DictConfig, OmegaConf


def setup_wandb(cfg: DictConfig, job_id: str | None = None) -> bool:
    """Initialize wandb with offline support for HPC.

    Args:
        cfg: Hydra configuration
        job_id: PBS job ID (if running on MetaCentrum)

    Returns:
        True if wandb is initialized successfully
    """
    if not cfg.logging.use_wandb:
        return False

    # Check if we're on a compute node (no internet)
    has_internet = _check_internet()

    # Set wandb directory from environment or config
    wandb_dir = os.environ.get("WANDB_DIR", "./wandb")
    os.makedirs(wandb_dir, exist_ok=True)

    # Build run name from config
    run_name = f"{cfg.model.family}_{cfg.model.name}"
    if job_id:
        run_name = f"{run_name}_{job_id.split('.')[0]}"

    # Configuration dict
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    try:
        wandb.init(
            project=cfg.project.name,
            name=run_name,
            config=dict(config_dict),  # type: ignore
            dir=wandb_dir,
            mode="online" if has_internet else "offline",
            save_code=True,
        )
        return True
    except Exception as e:
        print(f"[WARNING] Failed to initialize wandb: {e}")
        return False


def _check_internet() -> bool:
    """Check if internet is available (for wandb online mode)."""
    import socket

    try:
        socket.create_connection(("api.wandb.ai", 443), timeout=2)
        return True
    except OSError:
        return False


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log metrics to wandb if available."""
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def finish_wandb() -> None:
    """Finish wandb run if active."""
    if wandb.run is not None:
        wandb.finish()


def get_wandb_sync_dir(artifact_dir: str | Path | None = None) -> Path:
    """Get the wandb offline sync directory.

    Args:
        artifact_dir: Base artifact directory (defaults to WANDB_DIR env var)

    Returns:
        Path to wandb runs directory for syncing
    """
    if artifact_dir is None:
        artifact_dir = os.environ.get("WANDB_DIR", "./wandb")

    wandb_path = Path(artifact_dir)

    # Wandb offline runs are stored in wandb/offline-run-*/
    if (wandb_path / "offline-run-").exists():
        return wandb_path

    return wandb_path
