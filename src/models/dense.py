"""Static dense neural receiver for single-user SIMO 1x4."""

from __future__ import annotations

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ConvBlock(nn.Module):
    """Simple convolution block with optional activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str | None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        self.activation = _activation(activation) if activation is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))


class ResidualBottleneckBlock(nn.Module):
    """Residual bottleneck block with local 3x3 mixing."""

    def __init__(self, channels: int, hidden_dim: int, activation: str):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, hidden_dim, kernel_size=1, activation=activation),
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, activation=activation),
            ConvBlock(hidden_dim, channels, kernel_size=1, activation=None),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class StaticDenseNRX(nn.Module):
    """Static whole-grid CNN baseline for a single-user SIMO neural receiver."""

    def __init__(
        self,
        input_channels: int = 16,
        output_bits: int | None = 7168,
        state_dim: int = 56,
        num_cnn_blocks: int = 8,
        stem_hidden_dims: list[int] | None = None,
        block_hidden_dim: int = 48,
        readout_hidden_dim: int = 128,
        activation: str = "relu",
        num_subcarriers: int = 128,
        num_ofdm_symbols: int = 14,
        bits_per_symbol: int = 4,
        num_rx_antennas: int = 4,
        pilot_symbol_indices: list[int] | None = None,
        pilot_subcarrier_spacing: int = 2,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        stem_hidden_dims = stem_hidden_dims or [64, 64]
        pilot_symbol_indices = pilot_symbol_indices or [2, 11]

        expected_output_bits = bits_per_symbol * num_subcarriers * num_ofdm_symbols
        if output_bits is not None and output_bits != expected_output_bits:
            raise ValueError(f"output_bits={output_bits} does not match grid dimensions ({expected_output_bits})")

        self.input_channels = input_channels
        self.num_rx_antennas = num_rx_antennas
        self.num_subcarriers = num_subcarriers
        self.num_ofdm_symbols = num_ofdm_symbols
        self.pilot_symbol_indices = pilot_symbol_indices
        self.pilot_subcarrier_spacing = pilot_subcarrier_spacing

        stem_layers: list[nn.Module] = []
        current_channels = input_channels + 2
        for hidden_dim in stem_hidden_dims:
            stem_layers.append(ConvBlock(current_channels, hidden_dim, kernel_size=3, activation=activation))
            current_channels = hidden_dim
        stem_layers.append(ConvBlock(current_channels, state_dim, kernel_size=3, activation=None))
        self.stem = nn.Sequential(*stem_layers)

        self.backbone = nn.Sequential(
            *[ResidualBottleneckBlock(state_dim, block_hidden_dim, activation) for _ in range(num_cnn_blocks)]
        )
        self.readout_llrs = nn.Sequential(
            nn.Conv2d(state_dim, readout_hidden_dim, kernel_size=1),
            _activation(activation),
            nn.Conv2d(readout_hidden_dim, bits_per_symbol, kernel_size=1),
        )
        self.readout_channel = nn.Sequential(
            nn.Conv2d(state_dim, readout_hidden_dim, kernel_size=1),
            _activation(activation),
            nn.Conv2d(readout_hidden_dim, 2 * num_rx_antennas, kernel_size=1),
        )

        self.register_buffer(
            "pilot_distance_features",
            self._build_pilot_distance(
                num_subcarriers=self.num_subcarriers,
                num_ofdm_symbols=self.num_ofdm_symbols,
                pilot_symbol_indices=self.pilot_symbol_indices,
                pilot_subcarrier_spacing=self.pilot_subcarrier_spacing,
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if x.ndim != 4:
            raise ValueError(f"Expected input with shape [batch, channels, freq, time], got {tuple(x.shape)}")
        if x.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {x.shape[1]}")

        x = self._normalize_inputs(x)
        pilot_distance = self.pilot_distance_features.to(dtype=x.dtype).expand(x.shape[0], -1, -1, -1)

        state = self.stem(torch.cat([x, pilot_distance], dim=1))
        state = self.backbone(state)

        logits = self.readout_llrs(state).flatten(start_dim=1)
        channel_estimate = self.readout_channel(state)
        return {
            "logits": logits,
            "channel_estimate": channel_estimate,
            "intermediate_logits": [],
            "intermediate_channel_estimates": [],
        }

    def _normalize_inputs(self, x: torch.Tensor) -> torch.Tensor:
        received = x[:, : 2 * self.num_rx_antennas]
        channel_estimate = x[:, 2 * self.num_rx_antennas :]
        scale = received.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-6)
        normalized_received = received / scale
        normalized_channel = channel_estimate / scale
        return torch.cat([normalized_received, normalized_channel], dim=1)

    @staticmethod
    def _build_pilot_distance(
        num_subcarriers: int,
        num_ofdm_symbols: int,
        pilot_symbol_indices: list[int],
        pilot_subcarrier_spacing: int,
    ) -> torch.Tensor:
        freq_indices = torch.arange(num_subcarriers, dtype=torch.float32)
        time_indices = torch.arange(num_ofdm_symbols, dtype=torch.float32)
        pilot_subcarriers = torch.arange(0, num_subcarriers, pilot_subcarrier_spacing, dtype=torch.float32)
        pilot_symbols = torch.tensor(sorted(set(pilot_symbol_indices)), dtype=torch.float32)

        freq_distance = (freq_indices[:, None] - pilot_subcarriers[None, :]).abs().amin(dim=1)
        time_distance = (time_indices[:, None] - pilot_symbols[None, :]).abs().amin(dim=1)

        freq_distance = (freq_distance / max(float(pilot_subcarrier_spacing), 1.0)).view(1, 1, num_subcarriers, 1)
        time_distance = (time_distance / max(float(num_ofdm_symbols - 1), 1.0)).view(1, 1, 1, num_ofdm_symbols)
        return torch.cat(
            [
                freq_distance.expand(1, 1, num_subcarriers, num_ofdm_symbols),
                time_distance.expand(1, 1, num_subcarriers, num_ofdm_symbols),
            ],
            dim=1,
        )
