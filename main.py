"""Main entry point for training."""

import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.data import build_train_loader
from src.training import Trainer

os.environ["TF_USE_LEGACY_KERAS"] = "1"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the dense baseline training pipeline."""
    _set_seed(int(cfg.runtime.seed))
    print(f"Config: {cfg.model.name} ({cfg.model.family})")

    train_loader = build_train_loader(cfg)
    trainer = Trainer(cfg, job_id=os.environ.get("PBS_JOBID"))

    try:
        trainer.train(train_loader=train_loader, num_steps=int(cfg.training.max_steps))
        checkpoint_path = Path(cfg.training.checkpoint_dir) / f"{cfg.model.name}.pt"
        trainer.save_checkpoint(
            str(checkpoint_path),
            artifact_aliases=["latest", "final", f"step-{trainer.global_step}"],
            artifact_primary_alias="latest",
            log_artifact=True,
            checkpoint_kind="final",
        )
        print(f"Saved checkpoint to {checkpoint_path}")
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
