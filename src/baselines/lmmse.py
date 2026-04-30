"""Classical LS-MRC linear receiver baseline for 5G NRX.

Pipeline (per (subcarrier, OFDM-symbol) cell):
  1. Take pilot-derived LS channel estimate H_ls (already in input).
  2. MRC equalization across the 4 receive antennas:
        z = (h^H * y) / ||h||^2
     This is the classical maximum-ratio combiner for SIMO single-stream.
  3. Max-log soft demodulation for 16-QAM with natural-binary mapping
     (matching the data generator in src/data/sionna_generator.py):
        bits[0:2] -> in-phase  PAM-4 levels {-3,-1,+1,+3}/sqrt(10)
        bits[2:4] -> quadrature PAM-4 levels {-3,-1,+1,+3}/sqrt(10)
     Output bit logits in BCEWithLogitsLoss convention:
        logit(b) = log P(b=1) - log P(b=0)
        positive logit -> bit more likely to be 1.

Why this matches what every NRX paper compares against:
  - Channel estimation = pilot-based LS (the simplest standard).
  - Detection         = MRC + ML demodulation per cell (no neural net).
  - Uses true SNR     = standard "best-case classical" assumption (real
    receivers estimate SNR from pilots; we assume known SNR for fairness).

This file has zero trainable parameters and is intentionally purely
functional: vectorised PyTorch ops on GPU.
"""

from __future__ import annotations

import math

import torch

_PAM4_LEVELS_INT = (-3.0, -1.0, 1.0, 3.0)
_PAM4_NORM = math.sqrt(10.0)


def _split_input(x: torch.Tensor, num_rx_antennas: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the 16-channel input into received signal and LS channel estimate.

    Layout (matches src/data/sionna_generator._compose_resource_grid):
      channels [0 : N]       -> rx real
      channels [N : 2N]      -> rx imag
      channels [2N : 3N]     -> LS estimate real
      channels [3N : 4N]     -> LS estimate imag
    where N = num_rx_antennas.
    """
    n = num_rx_antennas
    if x.shape[1] != 4 * n:
        raise ValueError(f"Expected {4 * n} input channels, got {x.shape[1]}")
    rx = torch.complex(x[:, 0:n], x[:, n : 2 * n])
    h_ls = torch.complex(x[:, 2 * n : 3 * n], x[:, 3 * n : 4 * n])
    return rx, h_ls


def _pam4_max_log_llr(z_axis: torch.Tensor, post_eq_noise_var: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Max-log LLR for the 2 bits encoding one PAM-4 axis (in-phase or quadrature).

    Mapping (natural binary, matches generator):
        bit0 \in {0,1}, bit1 \in {0,1}; level index = bit0*2 + bit1
        levels in order: [-3, -1, +1, +3] / sqrt(10)

    Args:
        z_axis: real-valued equalised symbol component, shape (batch, freq, time).
        post_eq_noise_var: post-equalization noise variance, shape (batch, freq, time).
        levels: 4 PAM-4 levels (already normalised), shape (4,).

    Returns:
        Tensor (batch, freq, time, 2) of LLRs in BCE-with-logits convention:
        logit(b) > 0 means bit is more likely 1.
    """
    z_exp = z_axis.unsqueeze(-1)  # (batch, freq, time, 1)
    levels_b = levels.view(1, 1, 1, 4)  # broadcast across batch/freq/time
    distances = (z_exp - levels_b) ** 2  # (batch, freq, time, 4)

    # bit0 = MSB:  =0 -> indices {0,1}, =1 -> indices {2,3}
    d0_b0_eq_0 = distances[..., 0:2].amin(dim=-1)
    d0_b0_eq_1 = distances[..., 2:4].amin(dim=-1)

    # bit1 = LSB:  =0 -> indices {0,2}, =1 -> indices {1,3}
    d1_b1_eq_0 = torch.minimum(distances[..., 0], distances[..., 2])
    d1_b1_eq_1 = torch.minimum(distances[..., 1], distances[..., 3])

    inv_var = 1.0 / torch.clamp(post_eq_noise_var, min=1e-6)

    # logit(b) = log P(b=1) - log P(b=0) = (1/sigma^2) * [ min{d|b=0} - min{d|b=1} ]
    logit_b0 = inv_var * (d0_b0_eq_0 - d0_b0_eq_1)
    logit_b1 = inv_var * (d1_b1_eq_0 - d1_b1_eq_1)
    return torch.stack([logit_b0, logit_b1], dim=-1)


def lmmse_forward(
    inputs: torch.Tensor,
    snr_db: torch.Tensor,
    num_rx_antennas: int = 4,
    mode: str = "ls_mrc",
    channel_target: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Run a classical baseline receiver on a batch.

    Args:
        inputs: (batch, 16, freq, time) — same layout as the dataset's `inputs`
                tensor produced by SionnaNRXSimulator.
        snr_db: (batch,) per-sample SNR in dB. Used to compute the noise
                variance for soft demodulation.
        num_rx_antennas: Number of receive antennas (default 4 for SIMO 1x4).
        mode: which classical receiver to run:
              - "ls_mrc"    : LS channel estimate + MRC across antennas (realistic).
              - "genie_mrc" : true channel + MRC (upper bound; needs channel_target).
              - "single_ant": LS channel + single-antenna detection (naive lower bound,
                              uses only the first receive antenna — no diversity gain).
        channel_target: (batch, 2*N, freq, time) ground-truth channel (real/imag stacked),
                        required when mode == "genie_mrc".

    Returns:
        Dict matching the interface used by evaluate.py:
          - "logits": (batch, bits_per_symbol * freq * time) flattened bit logits
          - "channel_estimate": (batch, 2*num_rx_antennas, freq, time)
          - other fields stubbed to None / zeros
    """
    device = inputs.device
    dtype = inputs.dtype

    rx, h_ls = _split_input(inputs, num_rx_antennas)
    n = num_rx_antennas

    # Pick the channel used for detection, depending on mode
    if mode == "genie_mrc":
        if channel_target is None:
            raise ValueError("genie_mrc requires channel_target")
        if channel_target.shape[1] != 2 * n:
            raise ValueError(f"channel_target must have {2 * n} channels, got {channel_target.shape[1]}")
        h_used = torch.complex(channel_target[:, 0:n], channel_target[:, n : 2 * n])
    else:
        h_used = h_ls

    if mode == "single_ant":
        # Naive baseline: keep only the first receive antenna — discard 4× diversity gain.
        rx = rx[:, 0:1]
        h_used = h_used[:, 0:1]
    # Per-antenna squared magnitude of channel used for detection
    h_norm_sq = (h_used.real**2 + h_used.imag**2).sum(dim=1)  # (batch, freq, time)

    # MRC equaliser: z = (sum_n h_n^* * y_n) / sum_n |h_n|^2
    # For SIMO 1x4 single-stream this is mathematically identical to ZF.
    numerator = torch.sum(torch.conj(h_used) * rx, dim=1)  # (batch, freq, time)
    z = numerator / torch.clamp(h_norm_sq, min=1e-6)

    # Per-sample complex-noise variance: var(n) = 1 / SNR_linear   (matches generator)
    snr_linear = 10.0 ** (snr_db.to(device=device, dtype=dtype) / 10.0)
    sigma_sq = (1.0 / snr_linear).view(-1, 1, 1)  # broadcast over freq, time

    # Post-MRC effective noise variance: sigma^2 / sum_n |h_n|^2
    post_eq_noise_var = sigma_sq / torch.clamp(h_norm_sq, min=1e-6)

    levels = torch.tensor(_PAM4_LEVELS_INT, device=device, dtype=dtype) / _PAM4_NORM

    llr_iq_inphase = _pam4_max_log_llr(z.real, post_eq_noise_var, levels)
    llr_iq_quadr = _pam4_max_log_llr(z.imag, post_eq_noise_var, levels)

    # Stack to bit-level logits: bits 0,1 from in-phase axis; bits 2,3 from quadrature
    logits = torch.stack(
        [
            llr_iq_inphase[..., 0],
            llr_iq_inphase[..., 1],
            llr_iq_quadr[..., 0],
            llr_iq_quadr[..., 1],
        ],
        dim=1,
    )  # (batch, 4, freq, time)

    # The classical baseline does not refine the channel estimate — return LS as-is
    channel_estimate = torch.cat([h_ls.real, h_ls.imag], dim=1)

    zero = torch.tensor(0.0, device=device)
    return {
        "logits": logits.flatten(start_dim=1),
        "channel_estimate": channel_estimate,
        "intermediate_logits": [],
        "intermediate_channel_estimates": [],
        "router_logits": None,
        "router_probs": None,
        "routing_weights": None,
        "selected_expert_index": None,
        "expected_flops": zero,
        "realized_flops": zero,
        "expected_flops_ratio": zero,
        "realized_flops_ratio": zero,
        "expert_flops": zero,
    }


class LMMSEReceiver(torch.nn.Module):
    """nn.Module wrapper so the classical baseline composes with model-eval code.

    No trainable parameters. The forward signature requires the per-sample SNR
    in dB because the LLR computation needs the noise variance — a real receiver
    would estimate SNR from pilots; we use the generator's true SNR as the
    standard "perfect-SNR-knowledge" classical baseline.
    """

    def __init__(
        self,
        input_channels: int = 16,
        num_rx_antennas: int = 4,
        bits_per_symbol: int = 4,
        num_subcarriers: int = 128,
        num_ofdm_symbols: int = 14,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.num_rx_antennas = num_rx_antennas
        self.bits_per_symbol = bits_per_symbol
        self.num_subcarriers = num_subcarriers
        self.num_ofdm_symbols = num_ofdm_symbols

    @property
    def expert_names(self) -> list[str]:
        return []

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return lmmse_forward(x, snr_db, num_rx_antennas=self.num_rx_antennas)
