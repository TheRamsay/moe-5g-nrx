"""Cached dataset for loading pre-generated validation/test data.

Supports two backends:
- Local .pt files (legacy, loads everything into memory)
- HuggingFace datasets (lazy, memory-mapped via Arrow)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .constants import TRAIN_ITERATION_SEED_OFFSET


@dataclass(frozen=True)
class CachedNRXSample:
    """Single sample from a cached dataset."""

    inputs: torch.Tensor  # (channels, freq, time)
    bit_labels: torch.Tensor  # (bits_per_symbol, freq, time)
    channel_target: torch.Tensor  # (2*num_rx_antennas, freq, time)
    snr_db: float
    channel_profile: str
    modulation: str


@dataclass(frozen=True)
class CachedNRXBatch:
    """Batch of samples from a cached dataset."""

    inputs: torch.Tensor  # (batch, channels, freq, time)
    bit_labels: torch.Tensor  # (batch, bits_per_symbol, freq, time)
    channel_target: torch.Tensor  # (batch, 2*num_rx_antennas, freq, time)
    snr_db: torch.Tensor  # (batch,)
    channel_profile: str
    modulation: str

    def pin_memory(self) -> CachedNRXBatch:
        """Enable DataLoader pin_memory support for this custom batch type."""

        return CachedNRXBatch(
            inputs=self.inputs.pin_memory(),
            bit_labels=self.bit_labels.pin_memory(),
            channel_target=self.channel_target.pin_memory(),
            snr_db=self.snr_db.pin_memory(),
            channel_profile=self.channel_profile,
            modulation=self.modulation,
        )


class CachedNRXDataset(Dataset[CachedNRXSample]):
    """PyTorch MapDataset backed by a .pt file (all samples in memory)."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_samples: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Cached dataset not found: {self.path}")

        data = torch.load(self.path, map_location=device, weights_only=False)
        required_keys = {"inputs", "bit_labels", "channel_target", "snr_db"}
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(f"Cached dataset missing required keys: {missing}")

        self.inputs = data["inputs"]
        self.bit_labels = data["bit_labels"]
        self.channel_target = data["channel_target"]
        self.snr_db = data["snr_db"]

        metadata = data.get("metadata", {})
        self._channel_profile = metadata.get("channel_profile", self.path.stem)
        self._modulation = metadata.get("modulation", "qam16")

        if max_samples is not None and max_samples < len(self.inputs):
            self.inputs = self.inputs[:max_samples]
            self.bit_labels = self.bit_labels[:max_samples]
            self.channel_target = self.channel_target[:max_samples]
            self.snr_db = self.snr_db[:max_samples]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> CachedNRXSample:
        return CachedNRXSample(
            inputs=self.inputs[idx],
            bit_labels=self.bit_labels[idx],
            channel_target=self.channel_target[idx],
            snr_db=float(self.snr_db[idx]),
            channel_profile=self._channel_profile,
            modulation=self._modulation,
        )

    @property
    def num_samples(self) -> int:
        return len(self.inputs)

    @property
    def channel_profile(self) -> str:
        return self._channel_profile

    @property
    def modulation(self) -> str:
        return self._modulation


class HuggingFaceNRXDataset(Dataset[CachedNRXSample]):
    """PyTorch MapDataset backed by a HuggingFace dataset (lazy, memory-mapped)."""

    def __init__(
        self,
        repo_id: str,
        config: str,
        split: str,
        *,
        max_samples: int | None = None,
    ) -> None:
        from datasets import load_dataset

        ds = load_dataset(repo_id, config, split=split)
        if max_samples is not None and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        self._ds = ds.with_format("torch")
        self._channel_profile = config

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> CachedNRXSample:
        row = self._ds[idx]
        return CachedNRXSample(
            inputs=row["inputs"],
            bit_labels=row["bit_labels"],
            channel_target=row["channel_target"],
            snr_db=float(row["snr_db"]),
            channel_profile=self._channel_profile,
            modulation="qam16",
        )

    @property
    def num_samples(self) -> int:
        return len(self._ds)

    @property
    def channel_profile(self) -> str:
        return self._channel_profile

    @property
    def modulation(self) -> str:
        return "qam16"


class LocalArrowNRXDataset(Dataset[CachedNRXSample]):
    """PyTorch MapDataset backed by a local Arrow dataset directory."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_samples: int | None = None,
    ) -> None:
        from datasets import load_from_disk

        self.path = Path(path)
        if not self.path.exists() or not self.path.is_dir():
            raise FileNotFoundError(f"Local Arrow dataset directory not found: {self.path}")

        ds = load_from_disk(str(self.path))
        required_columns = {"inputs", "bit_labels", "channel_target", "snr_db"}
        missing = required_columns - set(ds.column_names)
        if missing:
            raise ValueError(f"Arrow dataset missing required columns: {missing}")

        if max_samples is not None and max_samples < len(ds):
            ds = ds.select(range(max_samples))

        self._ds = ds.with_format("torch")
        self._channel_profile = self.path.name
        self._modulation = "qam16"
        if len(self._ds) > 0:
            first_row = self._ds[0]
            profile_value = first_row.get("channel_profile")
            modulation_value = first_row.get("modulation")
            if profile_value is not None:
                self._channel_profile = str(profile_value)
            if modulation_value is not None:
                self._modulation = str(modulation_value)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> CachedNRXSample:
        row = self._ds[idx]
        return CachedNRXSample(
            inputs=row["inputs"],
            bit_labels=row["bit_labels"],
            channel_target=row["channel_target"],
            snr_db=float(row["snr_db"]),
            channel_profile=self._channel_profile,
            modulation=self._modulation,
        )

    @property
    def num_samples(self) -> int:
        return len(self._ds)

    @property
    def channel_profile(self) -> str:
        return self._channel_profile

    @property
    def modulation(self) -> str:
        return self._modulation


class PreloadedBatchedTrainDataset(IterableDataset[CachedNRXBatch]):
    """Batch-native iterable over an in-memory Arrow dataset (torch format).

    Expects the caller to have preloaded the dataset in the main process (see
    `train_loader._preload_dataset`). Each epoch shuffles indices with numpy
    — never copies the underlying Arrow table — and yields pre-batched tensors
    via `dataset[index_list]`, which is cheaper than per-sample `__getitem__`
    + `torch.stack` collation.

    Designed for `num_workers=0`: the 22 GB training Arrow table does not fit
    in RAM alongside multiple forked/spawned worker copies on a 32 GB node,
    and at ~3k samples/s the GPU is the bottleneck anyway.
    """

    def __init__(
        self,
        dataset,
        *,
        profile: str,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        base_seed: int | None = None,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self.profile = profile
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.base_seed = base_seed
        self._iteration_index = 0

    def __len__(self) -> int:
        n = len(self._dataset)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    @property
    def num_samples(self) -> int:
        return len(self._dataset)

    def _iteration_seed(self) -> int | None:
        if self.base_seed is None:
            return None
        return self.base_seed + self._iteration_index * TRAIN_ITERATION_SEED_OFFSET

    def __iter__(self):
        rng = np.random.RandomState(self._iteration_seed())
        n = len(self._dataset)
        indices = rng.permutation(n) if self.shuffle else np.arange(n)
        self._iteration_index += 1

        end = n - self.batch_size + 1 if self.drop_last else n
        for i in range(0, end, self.batch_size):
            chunk = indices[i : i + self.batch_size].tolist()
            batch = self._dataset[chunk]
            yield CachedNRXBatch(
                inputs=batch["inputs"],
                bit_labels=batch["bit_labels"],
                channel_target=batch["channel_target"],
                snr_db=batch["snr_db"].to(dtype=torch.float32),
                channel_profile=self.profile,
                modulation="qam16",
            )


def collate_cached_batch(samples: list[CachedNRXSample]) -> CachedNRXBatch:
    """Collate function for CachedNRXDataset and HuggingFaceNRXDataset."""
    return CachedNRXBatch(
        inputs=torch.stack([s.inputs for s in samples]),
        bit_labels=torch.stack([s.bit_labels for s in samples]),
        channel_target=torch.stack([s.channel_target for s in samples]),
        snr_db=torch.tensor([s.snr_db for s in samples], dtype=torch.float32),
        channel_profile=samples[0].channel_profile,
        modulation=samples[0].modulation,
    )


def build_cached_dataloader(
    path: str | Path | None = None,
    batch_size: int = 64,
    *,
    hf_repo: str | None = None,
    hf_config: str | None = None,
    hf_split: str | None = None,
    max_samples: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    """Build a DataLoader for a cached dataset (.pt, local Arrow dir, or HuggingFace).

    Provide either ``path`` for a local cache path (.pt file or Arrow directory), or
    ``hf_repo``/``hf_config``/``hf_split`` for a HuggingFace dataset.
    """
    if hf_repo is not None:
        dataset: Dataset[CachedNRXSample] = HuggingFaceNRXDataset(
            hf_repo,
            hf_config or "",
            hf_split or "train",
            max_samples=max_samples,
        )
    elif path is not None:
        path_obj = Path(path)
        if path_obj.is_dir():
            dataset = LocalArrowNRXDataset(path_obj, max_samples=max_samples)
        else:
            dataset = CachedNRXDataset(path_obj, max_samples=max_samples)
    else:
        raise ValueError("Provide either path (.pt or Arrow directory) or hf_repo (for HuggingFace)")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_cached_batch,
        drop_last=False,
    )
