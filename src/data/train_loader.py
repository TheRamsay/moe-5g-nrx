"""Training data loader construction."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

from .cached_dataset import HuggingFaceNRXBatchIterableDataset, collate_single_cached_batch
from .torch_dataset import build_dataloader


def _preload_dataset(hf_repo: str, profile: str, split: str, local_data_dir, max_samples):
    """Load a dataset in the main process so workers inherit it via fork+CoW."""
    if local_data_dir is not None:
        from datasets import load_from_disk

        ds = load_from_disk(str(Path(local_data_dir) / profile), keep_in_memory=True)
    else:
        from datasets import load_dataset

        ds = load_dataset(hf_repo, profile, split=split)
    if max_samples is not None and max_samples < len(ds):
        ds = ds.select(range(max_samples))
    return ds.with_format("torch")


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

    max_samples_int = int(max_samples) if max_samples is not None else None

    print(
        "[INFO] HF train loader: "
        f"repo={hf_repo} profiles={profiles} batch_size={batch_size} "
        f"workers={num_workers}x{len(profiles)} prefetch={prefetch_factor}"
        + (f" max_samples={max_samples_int}" if max_samples_int is not None else "")
        + (f" local_dir={local_data_dir}" if local_data_dir is not None else "")
    )

    # For local Arrow dirs: workers load their own shard independently (no CoW issues).
    # For HF hub: pre-load in main process so workers share via fork+CoW.
    preloaded = {}
    if local_data_dir is None:
        for profile in profiles:
            print(f"[INFO]   pre-loading {profile} from HF hub...", flush=True)
            preloaded[profile] = _preload_dataset(hf_repo, profile, "train", None, max_samples_int)
            n = len(preloaded[profile])
            print(f"[INFO]   {profile}: {n} samples, {n // batch_size} batches/epoch")
    else:
        for profile in profiles:
            n_est = max_samples_int or 50000
            print(f"[INFO]   {profile}: ~{n_est} samples, shard-per-worker from {local_data_dir}/{profile}/")

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
            max_samples=max_samples_int,
            local_data_dir=local_data_dir,
            preloaded_dataset=preloaded.get(profile),
        )
        loaders[profile] = DataLoader(
            ds,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_single_cached_batch,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            multiprocessing_context="spawn" if num_workers > 0 else None,
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
