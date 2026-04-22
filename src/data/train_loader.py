"""Training data loader construction."""

from __future__ import annotations

from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

from .cached_dataset import HuggingFaceNRXBatchIterableDataset, collate_single_cached_batch
from .torch_dataset import build_dataloader


class _AlternatingLoader:
    """Alternates batches between per-profile DataLoaders."""

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


class _TrainingBatchAdapter:
    """Convert NRXBatch/CachedNRXBatch into trainer tuple contract."""

    def __init__(self, dataloader):
        self._dataloader = dataloader

    def __iter__(self):
        for batch in self._dataloader:
            yield batch.inputs, batch.bit_labels.flatten(start_dim=1), batch.channel_target, batch.channel_profile


def _build_hf_loader(cfg: DictConfig, hf_repo: str):
    channel_profile = str(cfg.dataset.channel_profile).lower()
    batch_size = int(cfg.training.batch_size)
    if channel_profile == "mixed":
        profiles = list(cfg.dataset.mixed.get("profiles", ["uma", "tdlc"]))
    else:
        profiles = [channel_profile]

    num_workers = int(cfg.training.get("hf_num_workers", 2))
    prefetch_factor = int(cfg.training.get("hf_prefetch_factor", 1))
    max_samples = cfg.training.get("hf_max_samples")
    local_data_dir = cfg.training.get("hf_train_data_dir")

    print(
        "[INFO] HF train loader: "
        f"repo={hf_repo} profiles={profiles} batch_size={batch_size} "
        f"workers={num_workers}x{len(profiles)} prefetch={prefetch_factor}"
        + (f" max_samples={max_samples}" if max_samples is not None else "")
        + (f" local_dir={local_data_dir}" if local_data_dir is not None else "")
    )

    loaders = {}
    for profile in profiles:
        ds = HuggingFaceNRXBatchIterableDataset(
            hf_repo,
            profile,
            "train",
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            base_seed=int(cfg.runtime.seed),
            max_samples=int(max_samples) if max_samples is not None else None,
            local_data_dir=local_data_dir,
        )
        print(f"[INFO]   {profile}: {ds.num_samples} samples, {len(ds)} batches/epoch")
        loaders[profile] = DataLoader(
            ds,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_single_cached_batch,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

    if len(loaders) == 1:
        return next(iter(loaders.values()))
    return _AlternatingLoader(loaders)


def build_train_loader(cfg: DictConfig):
    """Build the training data loader from config, wrapped in the trainer tuple adapter."""
    with open_dict(cfg):
        cfg.data.num_subcarriers = int(cfg.dataset.num_subcarriers)
        cfg.data.num_ofdm_symbols = int(cfg.dataset.num_ofdm_symbols)

    hf_dataset = cfg.training.get("hf_dataset")
    loader = _build_hf_loader(cfg, str(hf_dataset)) if hf_dataset else build_dataloader(cfg)
    return _TrainingBatchAdapter(loader)
