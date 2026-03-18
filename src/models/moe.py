"""Mixture-of-Experts neural receiver."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Single expert network."""

    def __init__(self, input_channels: int, hidden_dim: int, depth: int, output_channels: int = 16):
        super().__init__()
        layers = []

        layers.append(nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.GELU())

        for _ in range(depth - 2):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.GELU())

        layers.append(nn.Conv2d(hidden_dim, output_channels, kernel_size=3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Router(nn.Module):
    """Channel-aware router for MoE."""

    def __init__(self, input_channels: int, num_experts: int = 3):
        super().__init__()
        self.num_experts = num_experts

        # Simple feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Linear(32, num_experts)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Route inputs to experts.

        Returns:
            weights: [B, num_experts] - routing weights
            indices: [B] - selected expert indices (top-1)
        """
        features = self.feature_extractor(x)
        features = features.flatten(1)
        logits = self.classifier(features)

        # Gumbel-softmax for differentiability
        weights = F.gumbel_softmax(logits, tau=temperature, hard=False)
        indices = logits.argmax(dim=-1)

        return weights, indices


class MoENRX(nn.Module):
    """Mixture-of-Experts neural receiver."""

    def __init__(
        self,
        input_channels: int = 16,
        output_bits: int = 7168,
        experts_config: dict | None = None,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_bits = output_bits
        self.temperature = temperature

        # Default expert configs
        if experts_config is None:
            experts_config = {
                "tiny": {"hidden_dim": 64, "depth": 2},
                "medium": {"hidden_dim": 128, "depth": 4},
                "heavy": {"hidden_dim": 256, "depth": 8},
            }

        # Router
        self.router = Router(input_channels, num_experts=len(experts_config))

        # Experts
        self.experts = nn.ModuleList(
            [Expert(input_channels, cfg["hidden_dim"], cfg["depth"]) for cfg in experts_config.values()]
        )

        # Output head (shared)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, output_bits)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with routing.

        Returns:
            logits: [B, output_bits]
            router_weights: [B, num_experts]
        """
        batch_size = x.size(0)

        # Get routing weights
        weights, indices = self.router(x, self.temperature)

        # Compute expert outputs (training: all experts, inference: top-1)
        if self.training:
            # Weighted combination of all experts
            expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
            # weights: [B, num_experts] -> [B, num_experts, 1, 1, 1]
            weights_expanded = weights.view(batch_size, -1, 1, 1, 1)
            x = (expert_outputs * weights_expanded).sum(dim=1)
        else:
            # Inference: use only top-1 expert (compute efficient)
            outputs = []
            for i in range(batch_size):
                expert_idx = indices[i]
                outputs.append(self.experts[expert_idx](x[i : i + 1]))
            x = torch.cat(outputs, dim=0)

        # Output head
        x = self.pool(x)
        x = x.flatten(1)
        logits = self.fc(x)

        return logits, weights
