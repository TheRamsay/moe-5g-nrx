"""Static dense neural receiver model."""

import torch
import torch.nn as nn


class StaticDenseNRX(nn.Module):
    """Static dense neural receiver.

    Simple CNN-based architecture for 5G signal demodulation.
    """

    def __init__(
        self,
        input_channels: int = 16,
        output_bits: int = 7168,  # 4 bits * 128 fft * 14 symbols
        hidden_dim: int = 256,
        depth: int = 8,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_bits = output_bits

        # Build backbone
        layers = []

        # First conv layer
        layers.append(nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.GELU())

        # Middle layers
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.GELU())

        # Last conv layer
        layers.append(nn.Conv2d(hidden_dim, 16, kernel_size=3, padding=1))

        self.backbone = nn.Sequential(*layers)

        # Global pooling and output layer
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, output_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, 16, F, T]

        Returns:
            logits: [B, output_bits]
        """
        x = self.backbone(x)
        x = self.pool(x)
        x = x.flatten(1)
        logits = self.fc(x)
        return logits
