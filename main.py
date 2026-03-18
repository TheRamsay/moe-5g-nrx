"""Main entry point for training - to be implemented."""

import os

import hydra
from omegaconf import DictConfig

os.environ["TF_USE_LEGACY_KERAS"] = "1"


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training function - to be implemented."""
    print(f"Config: {cfg.model.name} ({cfg.model.family})")

    # TODO: Create dataset and dataloader
    # TODO: Initialize trainer
    # TODO: Run training loop
    # TODO: Save checkpoint

    print("Training not yet implemented - structure only")


if __name__ == "__main__":
    main()
