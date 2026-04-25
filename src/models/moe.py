"""Mixture-of-Experts neural receiver model."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.compute import Conv2DShape, conv2d_flops, linear_flops

from .dense import ConvBlock, ResidualBottleneckBlock, _activation


class _ExpertHead(nn.Module):
    """One MoE expert branch with dense-compatible outputs."""

    def __init__(
        self,
        *,
        state_dim: int,
        num_cnn_blocks: int,
        block_hidden_dim: int,
        readout_hidden_dim: int,
        activation: str,
        bits_per_symbol: int,
        num_rx_antennas: int,
    ) -> None:
        super().__init__()
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

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.backbone(state)
        logits = self.readout_llrs(state)
        channel_estimate = self.readout_channel(state)
        return logits, channel_estimate


class MoENRX(nn.Module):
    """Shared-stem, channel-aware MoE neural receiver."""

    def __init__(
        self,
        input_channels: int = 16,
        output_bits: int | None = 7168,
        state_dim: int = 56,
        stem_hidden_dims: list[int] | None = None,
        activation: str = "relu",
        num_subcarriers: int = 128,
        num_ofdm_symbols: int = 14,
        bits_per_symbol: int = 4,
        num_rx_antennas: int = 4,
        pilot_symbol_indices: list[int] | None = None,
        pilot_subcarrier_spacing: int = 2,
        experts_config: dict[str, Any] | None = None,
        router_hidden_dim: int = 64,
        training_gate: str = "gumbel_softmax",
        inference_gate: str = "top1",
        temperature: float = 1.0,
        min_temperature: float = 0.5,
        use_input_statistics: bool = False,
        router_input_mode: str = "channel_aware",
        flops_normalizer: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs

        stem_hidden_dims = stem_hidden_dims or [64, 64]
        pilot_symbol_indices = pilot_symbol_indices or [2, 11]
        experts_config = experts_config or {
            "small": {"num_cnn_blocks": 8, "block_hidden_dim": 32, "readout_hidden_dim": 96},
            "mid": {"num_cnn_blocks": 8, "block_hidden_dim": 48, "readout_hidden_dim": 128},
            "large": {"num_cnn_blocks": 8, "block_hidden_dim": 64, "readout_hidden_dim": 128},
        }

        expected_output_bits = bits_per_symbol * num_subcarriers * num_ofdm_symbols
        if output_bits is not None and output_bits != expected_output_bits:
            raise ValueError(f"output_bits={output_bits} does not match grid dimensions ({expected_output_bits})")

        self.input_channels = input_channels
        self.state_dim = state_dim
        self.num_rx_antennas = num_rx_antennas
        self.num_subcarriers = num_subcarriers
        self.num_ofdm_symbols = num_ofdm_symbols
        self.bits_per_symbol = bits_per_symbol
        self.training_gate = training_gate
        self.inference_gate = inference_gate
        self.temperature = float(temperature)
        self.min_temperature = float(min_temperature)
        self.use_input_statistics = use_input_statistics
        if router_input_mode not in ("channel_aware", "random"):
            raise ValueError(f"router_input_mode must be 'channel_aware' or 'random', got {router_input_mode!r}")
        self.router_input_mode = router_input_mode
        self.flops_normalizer = max(float(flops_normalizer), 1.0)

        stem_layers: list[nn.Module] = []
        current_channels = input_channels + 2
        for hidden_dim in stem_hidden_dims:
            stem_layers.append(ConvBlock(current_channels, hidden_dim, kernel_size=3, activation=activation))
            current_channels = hidden_dim
        stem_layers.append(ConvBlock(current_channels, state_dim, kernel_size=3, activation=None))
        self.stem = nn.Sequential(*stem_layers)

        router_input_dim = 2 * state_dim + (3 if use_input_statistics else 0)
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, router_hidden_dim),
            _activation(activation),
            nn.Linear(router_hidden_dim, len(experts_config)),
        )

        self.expert_names = list(experts_config.keys())
        self.experts = nn.ModuleDict(
            OrderedDict(
                (
                    name,
                    _ExpertHead(
                        state_dim=state_dim,
                        num_cnn_blocks=int(config.get("num_cnn_blocks", 8)),
                        block_hidden_dim=int(config["block_hidden_dim"]),
                        readout_hidden_dim=int(config["readout_hidden_dim"]),
                        activation=activation,
                        bits_per_symbol=bits_per_symbol,
                        num_rx_antennas=num_rx_antennas,
                    ),
                )
                for name, config in experts_config.items()
            )
        )

        self.register_buffer(
            "pilot_distance_features",
            self._build_pilot_distance(
                num_subcarriers=num_subcarriers,
                num_ofdm_symbols=num_ofdm_symbols,
                pilot_symbol_indices=pilot_symbol_indices,
                pilot_subcarrier_spacing=pilot_subcarrier_spacing,
            ),
            persistent=False,
        )
        self.register_buffer(
            "expert_flops",
            torch.tensor(self._estimate_expert_flops(experts_config), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "base_flops",
            torch.tensor(
                self._estimate_base_flops(stem_hidden_dims, router_hidden_dim, router_input_dim), dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "max_flops",
            torch.tensor(float(self.base_flops.item() + self.expert_flops.max().item()), dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if x.ndim != 4:
            raise ValueError(f"Expected input with shape [batch, channels, freq, time], got {tuple(x.shape)}")
        if x.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {x.shape[1]}")

        x = self._normalize_inputs(x)
        pilot_distance = self.pilot_distance_features.to(dtype=x.dtype).expand(x.shape[0], -1, -1, -1)
        shared_state = self.stem(torch.cat([x, pilot_distance], dim=1))

        if self.router_input_mode == "random":
            # Channel-aware ablation: feed pure noise. Router cannot learn input-dependent
            # routing, so any heterogeneous behavior must come from architecture/losses,
            # not channel features. Test for the central "channel-aware routing" claim.
            router_input_dim = 2 * self.state_dim + (3 if self.use_input_statistics else 0)
            router_features_tensor = torch.randn(
                shared_state.shape[0], router_input_dim, device=shared_state.device, dtype=shared_state.dtype
            )
        else:
            router_features = [shared_state.mean(dim=(2, 3)), shared_state.amax(dim=(2, 3))]
            if self.use_input_statistics:
                router_features.append(self._input_statistics(x))
            router_features_tensor = torch.cat(router_features, dim=1)
        router_logits = self.router(router_features_tensor)
        router_probs = torch.softmax(router_logits, dim=-1)
        selected_expert_index = torch.argmax(router_probs, dim=-1)

        if self.training:
            routing_weights = self._training_routing_weights(router_logits)
            logits, channel_estimate = self._combine_all_experts(shared_state, routing_weights)
        else:
            routing_weights = F.one_hot(selected_expert_index, num_classes=len(self.expert_names)).to(
                dtype=shared_state.dtype
            )
            if self.inference_gate != "top1":
                raise ValueError(f"Unsupported inference gate: {self.inference_gate}")
            logits, channel_estimate = self._run_top1_inference(shared_state, selected_expert_index)

        expected_flops = self.base_flops + torch.sum(router_probs * self.expert_flops.to(router_probs.device), dim=-1)
        realized_flops = self.base_flops + self.expert_flops.to(router_probs.device)[selected_expert_index]
        normalized_denominator = self.max_flops.to(router_probs.device).clamp_min(self.flops_normalizer)

        return {
            "logits": logits.flatten(start_dim=1),
            "channel_estimate": channel_estimate,
            "intermediate_logits": [],
            "intermediate_channel_estimates": [],
            "router_logits": router_logits,
            "router_probs": router_probs,
            "routing_weights": routing_weights,
            "selected_expert_index": selected_expert_index,
            "expected_flops": expected_flops.mean(),
            "realized_flops": realized_flops.mean(),
            "expected_flops_ratio": expected_flops.mean() / normalized_denominator,
            "realized_flops_ratio": realized_flops.mean() / normalized_denominator,
            "expert_flops": self.expert_flops.to(router_probs.device),
        }

    def _combine_all_experts(
        self, shared_state: torch.Tensor, routing_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits: list[torch.Tensor] = []
        channels: list[torch.Tensor] = []
        for expert_name in self.expert_names:
            expert_logits, expert_channel = self.experts[expert_name](shared_state)
            logits.append(expert_logits)
            channels.append(expert_channel)

        logits_stack = torch.stack(logits, dim=1)
        channels_stack = torch.stack(channels, dim=1)
        weights = routing_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        combined_logits = torch.sum(logits_stack * weights, dim=1)
        combined_channel = torch.sum(channels_stack * weights, dim=1)
        return combined_logits, combined_channel

    def _run_top1_inference(
        self,
        shared_state: torch.Tensor,
        selected_expert_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = shared_state.shape[0]
        device = shared_state.device
        logits = torch.zeros(
            batch_size,
            self.bits_per_symbol,
            self.num_subcarriers,
            self.num_ofdm_symbols,
            device=device,
            dtype=shared_state.dtype,
        )
        channel_estimate = torch.zeros(
            batch_size,
            2 * self.num_rx_antennas,
            self.num_subcarriers,
            self.num_ofdm_symbols,
            device=device,
            dtype=shared_state.dtype,
        )
        for expert_index, expert_name in enumerate(self.expert_names):
            mask = selected_expert_index == expert_index
            if not torch.any(mask):
                continue
            expert_logits, expert_channel = self.experts[expert_name](shared_state[mask])
            logits[mask] = expert_logits
            channel_estimate[mask] = expert_channel
        return logits, channel_estimate

    def _training_routing_weights(self, router_logits: torch.Tensor) -> torch.Tensor:
        gate_name = self.training_gate.lower()
        temperature = max(self.min_temperature, self.temperature)
        if gate_name == "gumbel_softmax":
            return F.gumbel_softmax(router_logits, tau=temperature, hard=False, dim=-1)
        if gate_name == "softmax":
            return torch.softmax(router_logits / temperature, dim=-1)
        raise ValueError(f"Unsupported training gate: {self.training_gate}")

    def _input_statistics(self, normalized_inputs: torch.Tensor) -> torch.Tensor:
        received = normalized_inputs[:, : 2 * self.num_rx_antennas]
        channel_estimate = normalized_inputs[:, 2 * self.num_rx_antennas :]
        received_power = received.pow(2).mean(dim=(1, 2, 3))
        channel_power = channel_estimate.pow(2).mean(dim=(1, 2, 3))
        channel_variance = channel_estimate.var(dim=(1, 2, 3), unbiased=False)
        return torch.stack([received_power, channel_power, channel_variance], dim=1)

    def _normalize_inputs(self, x: torch.Tensor) -> torch.Tensor:
        received = x[:, : 2 * self.num_rx_antennas]
        channel_estimate = x[:, 2 * self.num_rx_antennas :]
        scale = received.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-6)
        normalized_received = received / scale
        normalized_channel = channel_estimate / scale
        return torch.cat([normalized_received, normalized_channel], dim=1)

    def _estimate_base_flops(self, stem_hidden_dims: list[int], router_hidden_dim: int, router_input_dim: int) -> float:
        flops = 0.0
        current_channels = self.input_channels + 2
        for hidden_dim in stem_hidden_dims:
            flops += conv2d_flops(
                Conv2DShape(
                    in_channels=current_channels,
                    out_channels=int(hidden_dim),
                    kernel_size=3,
                    height=self.num_subcarriers,
                    width=self.num_ofdm_symbols,
                )
            )
            current_channels = int(hidden_dim)
        flops += conv2d_flops(
            Conv2DShape(
                in_channels=current_channels,
                out_channels=self.state_dim,
                kernel_size=3,
                height=self.num_subcarriers,
                width=self.num_ofdm_symbols,
            )
        )
        flops += linear_flops(router_input_dim, router_hidden_dim)
        flops += linear_flops(router_hidden_dim, len(self.expert_names))
        return flops

    def _estimate_expert_flops(self, experts_config: dict[str, Any]) -> list[float]:
        expert_flops: list[float] = []
        for config in experts_config.values():
            num_cnn_blocks = int(config.get("num_cnn_blocks", 8))
            block_hidden_dim = int(config["block_hidden_dim"])
            readout_hidden_dim = int(config["readout_hidden_dim"])
            flops = 0.0
            for _ in range(num_cnn_blocks):
                flops += conv2d_flops(
                    Conv2DShape(
                        in_channels=self.state_dim,
                        out_channels=block_hidden_dim,
                        kernel_size=1,
                        height=self.num_subcarriers,
                        width=self.num_ofdm_symbols,
                    )
                )
                flops += conv2d_flops(
                    Conv2DShape(
                        in_channels=block_hidden_dim,
                        out_channels=block_hidden_dim,
                        kernel_size=3,
                        height=self.num_subcarriers,
                        width=self.num_ofdm_symbols,
                    )
                )
                flops += conv2d_flops(
                    Conv2DShape(
                        in_channels=block_hidden_dim,
                        out_channels=self.state_dim,
                        kernel_size=1,
                        height=self.num_subcarriers,
                        width=self.num_ofdm_symbols,
                    )
                )

            flops += conv2d_flops(
                Conv2DShape(
                    in_channels=self.state_dim,
                    out_channels=readout_hidden_dim,
                    kernel_size=1,
                    height=self.num_subcarriers,
                    width=self.num_ofdm_symbols,
                )
            )
            flops += conv2d_flops(
                Conv2DShape(
                    in_channels=readout_hidden_dim,
                    out_channels=self.bits_per_symbol,
                    kernel_size=1,
                    height=self.num_subcarriers,
                    width=self.num_ofdm_symbols,
                )
            )
            flops += conv2d_flops(
                Conv2DShape(
                    in_channels=self.state_dim,
                    out_channels=readout_hidden_dim,
                    kernel_size=1,
                    height=self.num_subcarriers,
                    width=self.num_ofdm_symbols,
                )
            )
            flops += conv2d_flops(
                Conv2DShape(
                    in_channels=readout_hidden_dim,
                    out_channels=2 * self.num_rx_antennas,
                    kernel_size=1,
                    height=self.num_subcarriers,
                    width=self.num_ofdm_symbols,
                )
            )
            expert_flops.append(flops)
        return expert_flops

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
