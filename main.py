"""Main entry point for training."""

import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data import NeuralReceiverDataset
from src.training import Trainer

os.environ["TF_USE_LEGACY_KERAS"] = "1"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_train_loader(cfg: DictConfig) -> DataLoader:
    required_samples = int(cfg.training.batch_size) * int(cfg.training.max_steps)
    dataset = NeuralReceiverDataset(
        num_samples=max(int(cfg.data.train_samples), required_samples),
        num_subcarriers=int(cfg.data.num_subcarriers),
        num_ofdm_symbols=int(cfg.data.num_ofdm_symbols),
        bits_per_symbol=int(cfg.data.bits_per_symbol),
        num_rx_antennas=int(cfg.data.num_rx_antennas),
        snr_db_range=[float(x) for x in cfg.data.snr_db_range],
        channel_estimator_noise_std=float(cfg.data.channel_estimator_noise_std),
        channel_variation=float(cfg.data.channel_variation),
        seed=int(cfg.runtime.seed),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=str(cfg.runtime.device).startswith("cuda"),
    )


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the dense baseline training pipeline."""
    _set_seed(int(cfg.runtime.seed))
    print(f"Config: {cfg.model.name} ({cfg.model.family})")

    train_loader = _build_train_loader(cfg)
    trainer = Trainer(cfg, job_id=os.environ.get("PBS_JOBID"))

    try:
        trainer.train(train_loader=train_loader, num_steps=int(cfg.training.max_steps))
        checkpoint_path = Path(cfg.training.checkpoint_dir) / f"{cfg.model.name}.pt"
        trainer.save_checkpoint(str(checkpoint_path))
        print(f"Saved checkpoint to {checkpoint_path}")
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
