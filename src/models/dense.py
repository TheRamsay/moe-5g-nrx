"""Static dense neural receiver model."""

import torch.nn as nn


class StaticDenseNRX(nn.Module):
    """Static dense neural receiver - to be implemented."""

    def __init__(self, input_channels: int = 16, output_bits: int = 7168, **kwargs):
        super().__init__()
        # TODO: Implement dense model architecture
        raise NotImplementedError("StaticDenseNRX not yet implemented")

    def forward(self, x):
        # TODO: Implement forward pass
        raise NotImplementedError("Forward pass not yet implemented")
