"""Main entry point for training."""

import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

from src.dataset import build_dataloader
from src.training import Trainer

os.environ["TF_USE_LEGACY_KERAS"] = "1"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_train_loader(cfg: DictConfig) -> DataLoader:
    # Keep model/data geometry synchronized with generator dataset config.
    with open_dict(cfg):
        cfg.data.num_subcarriers = int(cfg.dataset.num_subcarriers)
        cfg.data.num_ofdm_symbols = int(cfg.dataset.num_ofdm_symbols)

    return build_dataloader(cfg)


class _TrainingBatchAdapter:
    """Convert generator NRXBatch into trainer tuple contract."""

    def __init__(self, dataloader: DataLoader):
        self._dataloader = dataloader

    def __iter__(self):
        for batch in self._dataloader:
            yield batch.inputs, batch.bit_labels.flatten(start_dim=1), batch.channel_target


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the dense baseline training pipeline."""
    _set_seed(int(cfg.runtime.seed))
    print(f"Config: {cfg.model.name} ({cfg.model.family})")

    train_loader = _TrainingBatchAdapter(_build_train_loader(cfg))
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
