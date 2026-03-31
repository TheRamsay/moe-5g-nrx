"""Cached dataset for loading pre-generated validation/test data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    """PyTorch MapDataset for loading pre-generated validation/test data.

    Loads a single .pt file containing all samples into memory for fast access.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_samples: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Load a cached dataset from disk.

        Args:
            path: Path to the .pt file containing the cached dataset.
            max_samples: Optional limit on number of samples to use (for quick eval).
            device: Device to load tensors onto.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Cached dataset not found: {self.path}")

        data = torch.load(self.path, map_location=device, weights_only=False)
        self._validate_data(data)

        self.inputs = data["inputs"]
        self.bit_labels = data["bit_labels"]
        self.channel_target = data["channel_target"]
        self.snr_db = data["snr_db"]
        self.metadata = data["metadata"]

        if max_samples is not None and max_samples < len(self.inputs):
            self.inputs = self.inputs[:max_samples]
            self.bit_labels = self.bit_labels[:max_samples]
            self.channel_target = self.channel_target[:max_samples]
            self.snr_db = self.snr_db[:max_samples]

    def _validate_data(self, data: dict[str, Any]) -> None:
        required_keys = {"inputs", "bit_labels", "channel_target", "snr_db", "metadata"}
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(f"Cached dataset missing required keys: {missing}")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> CachedNRXSample:
        return CachedNRXSample(
            inputs=self.inputs[idx],
            bit_labels=self.bit_labels[idx],
            channel_target=self.channel_target[idx],
            snr_db=float(self.snr_db[idx]),
            channel_profile=self.metadata["channel_profile"],
            modulation=self.metadata["modulation"],
        )

    @property
    def num_samples(self) -> int:
        return len(self.inputs)

    @property
    def channel_profile(self) -> str:
        return self.metadata["channel_profile"]

    @property
    def modulation(self) -> str:
        return self.metadata["modulation"]


def collate_cached_batch(samples: list[CachedNRXSample]) -> CachedNRXBatch:
    """Collate function for CachedNRXDataset."""
    return CachedNRXBatch(
        inputs=torch.stack([s.inputs for s in samples]),
        bit_labels=torch.stack([s.bit_labels for s in samples]),
        channel_target=torch.stack([s.channel_target for s in samples]),
        snr_db=torch.tensor([s.snr_db for s in samples], dtype=torch.float32),
        channel_profile=samples[0].channel_profile,
        modulation=samples[0].modulation,
    )


def build_cached_dataloader(
    path: str | Path,
    batch_size: int,
    *,
    max_samples: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    """Build a DataLoader for a cached dataset.

    Args:
        path: Path to the .pt file containing the cached dataset.
        batch_size: Number of samples per batch.
        max_samples: Optional limit on number of samples (for quick eval).
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory for faster GPU transfer.
        shuffle: Whether to shuffle the dataset.

    Returns:
        DataLoader yielding CachedNRXBatch objects.
    """
    dataset = CachedNRXDataset(path, max_samples=max_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_cached_batch,
        drop_last=False,
    )
