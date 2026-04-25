"""Helpers for constructing models from Hydra or checkpoint configs."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from .dense import StaticDenseNRX
from .moe import MoENRX


def _section_get(section: DictConfig | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(section, DictConfig):
        return section.get(key, default)
    if isinstance(section, dict):
        return section.get(key, default)
    return default


def build_model_from_config(cfg: DictConfig | dict[str, Any]):
    """Construct a model from a resolved Hydra config or saved checkpoint config."""

    model_cfg = _section_get(cfg, "model", {})
    data_cfg = _section_get(cfg, "data", {})
    family = str(_section_get(model_cfg, "family", "static_dense"))

    common_kwargs = {
        "input_channels": int(_section_get(data_cfg, "input_channels", 16)),
        "output_bits": int(_section_get(data_cfg, "bits_per_symbol", 4))
        * int(_section_get(data_cfg, "num_subcarriers", 128))
        * int(_section_get(data_cfg, "num_ofdm_symbols", 14)),
        "num_subcarriers": int(_section_get(data_cfg, "num_subcarriers", 128)),
        "num_ofdm_symbols": int(_section_get(data_cfg, "num_ofdm_symbols", 14)),
        "bits_per_symbol": int(_section_get(data_cfg, "bits_per_symbol", 4)),
        "num_rx_antennas": int(_section_get(data_cfg, "num_rx_antennas", 4)),
    }

    if family == "static_dense":
        backbone_cfg = _section_get(model_cfg, "backbone", {})
        return StaticDenseNRX(
            state_dim=int(_section_get(backbone_cfg, "state_dim", 56)),
            num_cnn_blocks=int(_section_get(backbone_cfg, "num_cnn_blocks", 8)),
            stem_hidden_dims=list(_section_get(backbone_cfg, "stem_hidden_dims", [64, 64])),
            block_hidden_dim=int(_section_get(backbone_cfg, "block_hidden_dim", 48)),
            readout_hidden_dim=int(_section_get(backbone_cfg, "readout_hidden_dim", 128)),
            activation=str(_section_get(backbone_cfg, "activation", "relu")),
            pilot_symbol_indices=list(_section_get(backbone_cfg, "pilot_symbol_indices", [2, 11])),
            pilot_subcarrier_spacing=int(_section_get(backbone_cfg, "pilot_subcarrier_spacing", 2)),
            **common_kwargs,
        )

    if family == "moe":
        shared_stem_cfg = _section_get(model_cfg, "shared_stem", {})
        router_cfg = _section_get(model_cfg, "router", {})
        compute_cfg = _section_get(model_cfg, "compute", {})
        return MoENRX(
            state_dim=int(_section_get(shared_stem_cfg, "state_dim", 56)),
            stem_hidden_dims=list(_section_get(shared_stem_cfg, "stem_hidden_dims", [64, 64])),
            activation=str(_section_get(shared_stem_cfg, "activation", "relu")),
            pilot_symbol_indices=list(_section_get(shared_stem_cfg, "pilot_symbol_indices", [2, 11])),
            pilot_subcarrier_spacing=int(_section_get(shared_stem_cfg, "pilot_subcarrier_spacing", 2)),
            experts_config={
                name: cfg for name, cfg in dict(_section_get(model_cfg, "experts", {})).items() if cfg is not None
            },
            router_hidden_dim=int(_section_get(router_cfg, "hidden_dim", 64)),
            training_gate=str(_section_get(router_cfg, "training_gate", "gumbel_softmax")),
            inference_gate=str(_section_get(router_cfg, "inference_gate", "top1")),
            temperature=float(_section_get(router_cfg, "temperature", 1.0)),
            min_temperature=float(_section_get(router_cfg, "min_temperature", 0.5)),
            use_input_statistics=bool(_section_get(router_cfg, "use_input_statistics", False)),
            router_input_mode=str(_section_get(router_cfg, "input_mode", "channel_aware")),
            flops_normalizer=float(_section_get(compute_cfg, "flops_normalizer", 1.0)),
            **common_kwargs,
        )

    raise ValueError(f"Unsupported model family: {family}")
