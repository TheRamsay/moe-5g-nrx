import os
from collections.abc import Callable
from typing import cast

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.logging import finish_wandb, log_metrics, setup_wandb

os.environ["TF_USE_LEGACY_KERAS"] = "1"


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(f"Loaded config: {cfg.model.name} ({cfg.model.family})")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Setup wandb logging
    job_id = os.environ.get("PBS_JOBID")
    use_wandb = setup_wandb(cfg, job_id=job_id)

    if use_wandb:
        print(f"[INFO] Wandb initialized (offline={os.environ.get('WANDB_DIR', './wandb')})")
    else:
        print("[INFO] Wandb disabled or failed to initialize")

    try:
        # TODO: Your actual training code goes here
        # Example logging:
        for step in range(5):
            metrics = {
                "loss": 1.0 / (step + 1),
                "learning_rate": cfg.training.learning_rate,
            }
            log_metrics(metrics, step=step)
            print(f"Step {step}: loss={metrics['loss']:.4f}")

    finally:
        finish_wandb()
        print("[INFO] Run complete")


if __name__ == "__main__":
    cast("Callable[[], None]", main)()
