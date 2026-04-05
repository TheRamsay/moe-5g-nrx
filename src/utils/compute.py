"""Approximate compute accounting utilities.

The project compares models using consistent approximate FLOP counts rather than
hardware-specific kernel timings. The formulas here intentionally use a simple
multiply-add approximation for convolution and linear layers so dense and MoE
models share the same accounting rules.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Conv2DShape:
    in_channels: int
    out_channels: int
    kernel_size: int
    height: int
    width: int


def conv2d_flops(shape: Conv2DShape) -> float:
    """Approximate multiply-add FLOPs for one Conv2d invocation."""

    kernel_area = int(shape.kernel_size) * int(shape.kernel_size)
    return float(2 * shape.height * shape.width * shape.in_channels * shape.out_channels * kernel_area)


def linear_flops(in_features: int, out_features: int) -> float:
    """Approximate multiply-add FLOPs for one Linear layer invocation."""

    return float(2 * in_features * out_features)


def mixture_expected_flops(
    base_flops: float,
    expert_flops: list[float],
    routing_weights: list[float],
) -> float:
    """Expected FLOPs for a routed mixture given expert probabilities."""

    return float(base_flops + sum(weight * flops for weight, flops in zip(routing_weights, expert_flops, strict=True)))


def mixture_realized_flops(base_flops: float, expert_flops: list[float], selected_index: int) -> float:
    """Realized FLOPs when exactly one expert is executed."""

    return float(base_flops + expert_flops[selected_index])
