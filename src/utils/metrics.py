from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BatchErrorMetrics:
    """Bit-, block-, and symbol-level error tensors and rates for one batch."""

    bit_errors: torch.Tensor
    block_errors: torch.Tensor
    symbol_errors: torch.Tensor
    bit_errors_per_block: torch.Tensor
    ber: torch.Tensor
    bler: torch.Tensor
    ser: torch.Tensor


def compute_batch_error_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    bits_per_symbol: int,
    num_subcarriers: int,
    num_ofdm_symbols: int,
) -> BatchErrorMetrics:
    """Compute BER/BLER/SER from flattened per-bit logits and targets."""

    if logits.shape != targets.shape:
        raise ValueError(f"logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D flattened tensors [batch, bits], got {tuple(logits.shape)}")

    expected_bits = bits_per_symbol * num_subcarriers * num_ofdm_symbols
    if logits.shape[1] != expected_bits:
        raise ValueError(
            f"Flattened bit dimension {logits.shape[1]} does not match expected "
            f"{expected_bits} (= {bits_per_symbol}*{num_subcarriers}*{num_ofdm_symbols})"
        )

    predictions = (logits > 0.0).to(dtype=targets.dtype)
    bit_errors = (predictions != targets).to(dtype=torch.float32)

    batch_size = bit_errors.shape[0]
    bit_errors_grid = bit_errors.reshape(batch_size, bits_per_symbol, num_subcarriers, num_ofdm_symbols)
    symbol_errors = bit_errors_grid.any(dim=1).to(dtype=torch.float32)
    block_errors = bit_errors.any(dim=1).to(dtype=torch.float32)
    bit_errors_per_block = bit_errors.sum(dim=1)

    return BatchErrorMetrics(
        bit_errors=bit_errors,
        block_errors=block_errors,
        symbol_errors=symbol_errors,
        bit_errors_per_block=bit_errors_per_block,
        ber=bit_errors.mean(),
        bler=block_errors.mean(),
        ser=symbol_errors.mean(),
    )
