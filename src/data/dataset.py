"""Synthetic dataset for the dense 5G neural receiver baseline."""

from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset


class NeuralReceiverDataset(Dataset):
    """Generate on-the-fly OFDM grids with noisy channel estimates."""

    def __init__(
        self,
        num_samples: int = 10000,
        num_subcarriers: int = 128,
        num_ofdm_symbols: int = 14,
        bits_per_symbol: int = 4,
        num_rx_antennas: int = 4,
        snr_db_range: tuple[float, float] | list[float] = (0.0, 24.0),
        channel_estimator_noise_std: float = 0.15,
        channel_variation: float = 0.25,
        seed: int = 67,
        **kwargs,
    ):
        del kwargs

        if len(snr_db_range) != 2:
            raise ValueError("snr_db_range must contain [min_snr_db, max_snr_db]")

        self.num_samples = num_samples
        self.num_subcarriers = num_subcarriers
        self.num_ofdm_symbols = num_ofdm_symbols
        self.bits_per_symbol = bits_per_symbol
        self.num_rx_antennas = num_rx_antennas
        self.snr_db_range = (float(snr_db_range[0]), float(snr_db_range[1]))
        self.channel_estimator_noise_std = channel_estimator_noise_std
        self.channel_variation = channel_variation
        self.seed = seed
        self.gray_levels = torch.tensor([-3.0, -1.0, 3.0, 1.0], dtype=torch.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + idx)

        bits = torch.randint(
            0,
            2,
            (self.num_subcarriers, self.num_ofdm_symbols, self.bits_per_symbol),
            generator=generator,
            dtype=torch.int64,
        )
        tx_symbols = self._bits_to_qam(bits)
        channel = self._sample_channel(generator)

        snr_db = torch.empty(1).uniform_(*self.snr_db_range, generator=generator).item()
        snr_linear = 10 ** (snr_db / 10.0)
        noise_std = math.sqrt(1.0 / (2.0 * snr_linear))

        received = channel * tx_symbols.unsqueeze(0) + noise_std * self._complex_randn(channel.shape, generator)
        estimate_noise = (self.channel_estimator_noise_std + 0.35 * noise_std) * self._complex_randn(
            channel.shape, generator
        )
        ls_estimate = channel + estimate_noise

        features = torch.cat(
            [received.real, received.imag, ls_estimate.real, ls_estimate.imag],
            dim=0,
        ).to(dtype=torch.float32)
        target_bits = bits.permute(2, 0, 1).reshape(-1).to(dtype=torch.float32)
        channel_target = torch.cat([channel.real, channel.imag], dim=0).to(dtype=torch.float32)
        return features, target_bits, channel_target

    def _bits_to_qam(self, bits: torch.Tensor) -> torch.Tensor:
        i_idx = (bits[..., 0] * 2 + bits[..., 1]).to(dtype=torch.int64)
        q_idx = (bits[..., 2] * 2 + bits[..., 3]).to(dtype=torch.int64)
        i = self.gray_levels[i_idx]
        q = self.gray_levels[q_idx]
        return torch.complex(i, q) / math.sqrt(10.0)

    def _sample_channel(self, generator: torch.Generator) -> torch.Tensor:
        shape = (self.num_rx_antennas, self.num_subcarriers, self.num_ofdm_symbols)
        base = self._complex_randn((self.num_rx_antennas, 1, 1), generator)
        freq_profile = self._complex_randn((self.num_rx_antennas, self.num_subcarriers, 1), generator)
        time_profile = self._complex_randn((self.num_rx_antennas, 1, self.num_ofdm_symbols), generator)
        local_detail = self._complex_randn(shape, generator)

        channel = base + 0.35 * freq_profile + 0.25 * time_profile + self.channel_variation * local_detail
        rms = channel.abs().pow(2).mean().sqrt().clamp_min(1e-6)
        return channel / rms

    @staticmethod
    def _complex_randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
        real = torch.randn(shape, generator=generator, dtype=torch.float32)
        imag = torch.randn(shape, generator=generator, dtype=torch.float32)
        return torch.complex(real, imag) / math.sqrt(2.0)
