"""Cached dataset for loading pre-generated validation/test data.

Supports two backends:
- Local .pt files (legacy, loads everything into memory)
- HuggingFace datasets (lazy, memory-mapped via Arrow)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


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
    """Build a DataLoader for a cached dataset (.pt file or HuggingFace).

    Provide either ``path`` for a local .pt file, or ``hf_repo``/``hf_config``/``hf_split``
    for a HuggingFace dataset.
    """
    if hf_repo is not None:
        dataset: Dataset[CachedNRXSample] = HuggingFaceNRXDataset(
            hf_repo,
            hf_config or "",
            hf_split or "train",
            max_samples=max_samples,
        )
    elif path is not None:
        dataset = CachedNRXDataset(path, max_samples=max_samples)
    else:
        raise ValueError("Provide either path (for .pt) or hf_repo (for HuggingFace)")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_cached_batch,
        drop_last=False,
    )
