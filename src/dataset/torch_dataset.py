from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, IterableDataset

from .sionna_generator import SimulatorBatch, SionnaNRXSimulator, build_simulator_from_cfg


@dataclass(frozen=True)
class NRXBatch:
    """Torch-friendly batch container."""

    inputs: torch.Tensor  # (batch, channels, freq, time)
    bit_labels: torch.Tensor  # (batch, bits_per_symbol, freq, time)
    channel_target: torch.Tensor  # (batch, 2*num_rx_antennas, freq, time)
    snr_db: torch.Tensor  # (batch,)
    channel_profile: str
    modulation: str


class NRXIterableDataset(IterableDataset[NRXBatch]):
    """Wraps the simulator with a PyTorch IterableDataset API."""

    def __init__(
        self,
        simulator_cfg: dict[str, Any],
        batch_size: int,
        *,
        max_batches: int | None,
        base_seed: int | None,
    ) -> None:
        super().__init__()
        self.simulator_cfg = simulator_cfg
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.base_seed = base_seed
        self._iteration_index = 0
        self._simulator: SionnaNRXSimulator | None = None

    def _get_or_create_simulator(self) -> SionnaNRXSimulator:
        if self._simulator is None:
            cfg = OmegaConf.create(self.simulator_cfg)
            self._simulator = build_simulator_from_cfg(cfg)
        return self._simulator

    def __iter__(self) -> Iterator[NRXBatch]:
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        simulator = self._get_or_create_simulator()

        # Offset RNG streams by worker and iteration to avoid overlap.
        worker_offset = worker_id * 100_000
        iteration_offset = self._iteration_index * 1_000_000
        if self.base_seed is not None:
            simulator.reseed(self.base_seed + worker_offset + iteration_offset)
        self._iteration_index += 1

        if self.max_batches is None:
            while True:
                sim_batch = simulator.generate_batch(self.batch_size)
                yield self._to_torch(sim_batch)
            return

        # Shard global batch budget across workers: worker k produces
        # indices k, k+num_workers, k+2*num_workers, ...
        global_batch_idx = worker_id
        while global_batch_idx < self.max_batches:
            sim_batch = simulator.generate_batch(self.batch_size)
            yield self._to_torch(sim_batch)
            global_batch_idx += num_workers

    def _to_torch(self, sim_batch: SimulatorBatch) -> NRXBatch:
        inputs = torch.from_numpy(sim_batch.resource_grid)
        bit_labels = torch.from_numpy(sim_batch.bit_labels)
        channel_target = torch.from_numpy(sim_batch.channel_target)
        snr_db = torch.from_numpy(sim_batch.snr_db)
        metadata = sim_batch.metadata
        return NRXBatch(
            inputs=inputs,
            bit_labels=bit_labels,
            channel_target=channel_target,
            snr_db=snr_db,
            channel_profile=metadata["channel_profile"],
            modulation=metadata["modulation"],
        )


def _single_batch_collate(batch: NRXBatch | list[NRXBatch]) -> NRXBatch:
    """Handle both direct and list-style invocations from DataLoader."""

    if isinstance(batch, list):
        return batch[0]
    return batch


def build_dataloader(cfg: DictConfig) -> DataLoader:
    """Instantiate the NRX dataloader from a Hydra config."""

    batch_size = int(cfg.training.batch_size)
    max_batches_cfg = int(cfg.dataset.max_batches_per_epoch)
    max_batches = max_batches_cfg if max_batches_cfg > 0 else None
    simulator_cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(simulator_cfg, dict):
        raise TypeError("Resolved Hydra config must be a dictionary-like object.")

    dataset = NRXIterableDataset(
        simulator_cfg,
        batch_size=batch_size,
        max_batches=max_batches,
        base_seed=int(cfg.runtime.seed),
    )
    loader_cfg = cfg.dataset.loader
    num_workers = int(loader_cfg.num_workers)
    loader_kwargs: dict[str, Any] = {
        "batch_size": None,
        "num_workers": num_workers,
        "pin_memory": bool(loader_cfg.pin_memory),
        "persistent_workers": bool(loader_cfg.persistent_workers) and num_workers > 0,
        "collate_fn": _single_batch_collate,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(loader_cfg.prefetch_factor)
        # TensorFlow state in Sionna workers is safer with spawn than fork.
        loader_kwargs["multiprocessing_context"] = "spawn"
    return DataLoader(dataset, **loader_kwargs)


def peek_batch(dataloader: DataLoader) -> NRXBatch:
    """Fetch a single batch for quick sanity checks."""

    return next(iter(dataloader))
