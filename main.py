"""Main entry point for training."""

import os

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.dataset import NeuralReceiverDataset
from src.training.trainer import Trainer

os.environ["TF_USE_LEGACY_KERAS"] = "1"


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training function."""
    print(f"Starting training: {cfg.model.name} ({cfg.model.family})")

    # Set random seed
    torch.manual_seed(cfg.runtime.seed)

    # Get job ID if running on PBS
    job_id = os.environ.get("PBS_JOBID")

    # Create dataset
    print("[main] Creating dataset...")
    dataset = NeuralReceiverDataset(
        num_samples=cfg.training.max_steps * cfg.training.batch_size,
        batch_size=cfg.training.batch_size,
        num_rx_antennas=cfg.data.num_rx_antennas,
    )

    # Create dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,  # Keep it simple for now
    )

    # Create trainer
    print("[main] Initializing trainer...")
    trainer = Trainer(cfg, job_id=job_id)

    # Train
    try:
        trainer.train(train_loader, num_steps=cfg.training.max_steps)

        # Save checkpoint
        output_dir = os.environ.get("RUN_OUTPUT_DIR", "./outputs")
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "final_checkpoint.pt")
        trainer.save_checkpoint(checkpoint_path)

        print("[main] Training completed successfully!")

    except Exception as e:
        print(f"[main] Training failed: {e}")
        raise

    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
