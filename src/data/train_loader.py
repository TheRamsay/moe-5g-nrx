"""Training data loader construction.

Single code path: preload each profile's Arrow dataset into main-process RAM,
wrap it in a batch-native iterable, and iterate with `num_workers=0`.

Why single-process:
  * The 50k-sample Array3D training subset is ~22 GB in RAM. On the 32 GB
    cluster nodes there is no budget for additional forked/spawned worker
    copies of the table.
  * Observed throughput (~3k samples/s) is GPU-bound: `data_t ≈ 0` at steady
    state, so DataLoader workers would only add IPC overhead, not hide latency.
  * Earlier worker-based configurations deadlocked (fork + inherited TF/torch
    thread mutexes) or OOMed (spawn re-loading the dataset per worker, or CoW
    blowup from per-epoch Arrow shuffle copies). See git history.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

from .cached_dataset import CachedNRXBatch, PreloadedBatchedTrainDataset
from .torch_dataset import build_dataloader


def _preload_dataset(hf_repo: str, profile: str, split: str, local_data_dir, max_samples):
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
    """Alternates batches between per-profile DataLoaders (for `mixed` training)."""

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
    """Convert CachedNRXBatch into the trainer's (inputs, labels, target, profile) tuple."""

    def __init__(self, dataloader):
        self._dataloader = dataloader

    def __iter__(self):
        for batch in self._dataloader:
            yield batch.inputs, batch.bit_labels.flatten(start_dim=1), batch.channel_target, batch.channel_profile


def _identity_collate(batch: CachedNRXBatch | list[CachedNRXBatch]) -> CachedNRXBatch:
    """Pass-through collate: PreloadedBatchedTrainDataset already yields full batches."""
    return batch[0] if isinstance(batch, list) else batch


def _build_hf_loader(cfg: DictConfig, hf_repo: str):
    channel_profile = str(cfg.dataset.channel_profile).lower()
    batch_size = int(cfg.training.batch_size)
    if channel_profile == "mixed":
        profiles = list(cfg.dataset.mixed.get("profiles", ["uma", "tdlc"]))
    else:
        profiles = [channel_profile]

    max_samples = cfg.training.get("hf_max_samples")
    local_data_dir = cfg.training.get("hf_train_data_dir")
    max_samples_int = int(max_samples) if max_samples is not None else None

    print(
        f"[INFO] train loader: repo={hf_repo} profiles={profiles} batch_size={batch_size} "
        + (f"max_samples={max_samples_int} " if max_samples_int is not None else "")
        + (f"local_dir={local_data_dir}" if local_data_dir is not None else "source=hf_hub")
    )

    loaders = {}
    for profile in profiles:
        src = "local dir" if local_data_dir is not None else "HF hub"
        print(f"[INFO]   pre-loading {profile} from {src}...", flush=True)
        arrow_ds = _preload_dataset(hf_repo, profile, "train", local_data_dir, max_samples_int)
        n = len(arrow_ds)
        print(f"[INFO]   {profile}: {n} samples, {n // batch_size} batches/epoch")

        dataset = PreloadedBatchedTrainDataset(
            arrow_ds,
            profile=profile,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            base_seed=int(cfg.runtime.seed),
        )
        loaders[profile] = DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=True,
            collate_fn=_identity_collate,
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
