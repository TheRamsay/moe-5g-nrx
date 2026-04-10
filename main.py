"""Main entry point for training."""

import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

from src.data import HuggingFaceNRXDataset, build_dataloader, collate_cached_batch
from src.training import Trainer

os.environ["TF_USE_LEGACY_KERAS"] = "1"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _AlternatingLoader:
    """Alternates batches between per-profile DataLoaders (matches MixedNRXIterableDataset)."""

    def __init__(self, loaders: dict[str, DataLoader]):
        self.loaders = loaders
        self.profiles = list(loaders.keys())

    def __iter__(self):
        iters = {p: iter(dl) for p, dl in self.loaders.items()}
        idx = 0
        while True:
            profile = self.profiles[idx % len(self.profiles)]
            try:
                yield next(iters[profile])
            except StopIteration:
                iters[profile] = iter(self.loaders[profile])
                yield next(iters[profile])
            idx += 1


def _build_hf_train_loader(cfg: DictConfig, hf_repo: str):
    """Build a training loader backed by a HuggingFace dataset."""
    channel_profile = str(cfg.dataset.channel_profile).lower()
    batch_size = int(cfg.training.batch_size)

    if channel_profile == "mixed":
        profiles = list(cfg.dataset.mixed.get("profiles", ["uma", "tdlc"]))
    else:
        profiles = [channel_profile]

    num_workers = int(cfg.training.get("hf_num_workers", 4))
    prefetch_factor = int(cfg.training.get("hf_prefetch_factor", 4))

    loaders = {}
    for profile in profiles:
        ds = HuggingFaceNRXDataset(hf_repo, profile, "train")
        print(f"[INFO] HF training dataset: {hf_repo}/{profile}/train ({len(ds)} samples)")
        loaders[profile] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_cached_batch,
            drop_last=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

    if len(loaders) == 1:
        return next(iter(loaders.values()))
    return _AlternatingLoader(loaders)


def _build_train_loader(cfg: DictConfig):
    # Keep model/data geometry synchronized with generator dataset config.
    with open_dict(cfg):
        cfg.data.num_subcarriers = int(cfg.dataset.num_subcarriers)
        cfg.data.num_ofdm_symbols = int(cfg.dataset.num_ofdm_symbols)

    hf_dataset = cfg.training.get("hf_dataset")
    if hf_dataset:
        return _build_hf_train_loader(cfg, str(hf_dataset))

    return build_dataloader(cfg)


class _TrainingBatchAdapter:
    """Convert NRXBatch/CachedNRXBatch into trainer tuple contract."""

    def __init__(self, dataloader):
        self._dataloader = dataloader

    def __iter__(self):
        for batch in self._dataloader:
            yield batch.inputs, batch.bit_labels.flatten(start_dim=1), batch.channel_target, batch.channel_profile


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
