"""DeepMIMO-backed OOD dataset generation helpers."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Array3D, Dataset, Features, Value


@dataclass(frozen=True)
class DeepMIMOGenerationConfig:
    """Configuration for generating model-ready OOD samples from DeepMIMO channels."""

    scenario: str
    num_samples: int
    num_rx_antennas: int
    num_subcarriers: int
    num_ofdm_symbols: int
    bits_per_symbol: int
    snr_db_min: float
    snr_db_max: float
    estimation_noise_std: float
    pilot_spacing_freq: int
    pilot_spacing_time: tuple[int, ...]
    seed: int
    profile_name: str = "deepmimo"
    modulation: str = "qam16"
    active_users_only: bool = True
    dataset_folder: str = "./data/deepmimov3/"
    active_bs: tuple[int, ...] = (1,)
    user_rows: tuple[int, ...] | None = None
    user_subsampling: float = 1.0
    num_paths: int = 10
    use_all_rows_if_unspecified: bool = True


def _flatten_channel_blocks(channels: Any) -> list[np.ndarray]:
    if isinstance(channels, np.ndarray):
        return [channels]
    if isinstance(channels, list):
        blocks: list[np.ndarray] = []
        for item in channels:
            blocks.extend(_flatten_channel_blocks(item))
        return blocks
    return []


def _to_four_dim_channel_block(raw_block: np.ndarray) -> np.ndarray:
    block = np.asarray(raw_block)
    while block.ndim > 4:
        block = block[..., 0]
    if block.ndim != 4:
        raise ValueError(f"Expected channel block with 4 dimensions, got shape={block.shape}")
    return block


def _resolve_subcarrier_indices(total_subcarriers: int, target_subcarriers: int) -> np.ndarray:
    if total_subcarriers < target_subcarriers:
        raise ValueError(f"DeepMIMO channel has only {total_subcarriers} subcarriers, required {target_subcarriers}")
    if total_subcarriers == target_subcarriers:
        return np.arange(total_subcarriers, dtype=np.int64)
    return np.linspace(0, total_subcarriers - 1, num=target_subcarriers, dtype=np.int64)


def _extract_rx_subcarrier_channel(user_channel: np.ndarray, num_rx_antennas: int) -> np.ndarray:
    channel = np.asarray(user_channel)
    if channel.ndim == 2:
        channel = channel[:, None, :]
    if channel.ndim != 3:
        raise ValueError(f"Expected per-user channel tensor with 3 dims, got shape={channel.shape}")

    dim_a, dim_b, _ = channel.shape
    if max(dim_a, dim_b) < num_rx_antennas:
        raise ValueError(f"Unable to map channel shape {channel.shape} to {num_rx_antennas} receive antennas")

    if dim_a >= num_rx_antennas and dim_b == 1:
        rx_channel = channel[:num_rx_antennas, 0, :]
    elif dim_b >= num_rx_antennas and dim_a == 1:
        rx_channel = channel[0, :num_rx_antennas, :]
    elif dim_a >= num_rx_antennas:
        rx_channel = channel[:num_rx_antennas, 0, :]
    else:
        rx_channel = channel[0, :num_rx_antennas, :]

    return np.asarray(rx_channel, dtype=np.complex64)


def _normalize_channel_power(channel: np.ndarray) -> np.ndarray:
    power = np.mean(np.abs(channel) ** 2)
    power = max(float(power), 1e-12)
    return channel / np.sqrt(power)


def _pam4_from_bits(bits: np.ndarray) -> np.ndarray:
    bit0 = bits[0]
    bit1 = bits[1]
    indices = bit0 * 2 + bit1
    mapping = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
    return mapping[indices.astype(np.int8)]


def _modulate(bit_labels: np.ndarray) -> np.ndarray:
    inphase_bits = bit_labels[0:2, :, :]
    quadrature_bits = bit_labels[2:4, :, :]
    inphase_level = _pam4_from_bits(inphase_bits)
    quadrature_level = _pam4_from_bits(quadrature_bits)
    return (inphase_level + 1j * quadrature_level) / np.sqrt(10.0)


def _stamp_known_pilot_bits(
    bit_labels: np.ndarray,
    pilot_freq_idx: np.ndarray,
    pilot_time_idx: np.ndarray,
) -> None:
    for freq_idx in pilot_freq_idx:
        for time_idx in pilot_time_idx:
            bit_labels[:, freq_idx, time_idx] = 0.0


def _interp_complex_1d(
    known_x: np.ndarray,
    known_y: np.ndarray,
    target_x: np.ndarray,
) -> np.ndarray:
    if known_x.size == 1:
        return np.full(target_x.shape, known_y[0], dtype=np.complex64)
    interp_real = np.interp(target_x, known_x, known_y.real)
    interp_imag = np.interp(target_x, known_x, known_y.imag)
    return (interp_real + 1j * interp_imag).astype(np.complex64)


def _interpolate_pilot_grid(
    pilot_values: np.ndarray,
    pilot_freq_idx: np.ndarray,
    pilot_time_idx: np.ndarray,
    full_freq: int,
    full_time: int,
) -> np.ndarray:
    frequency_filled = np.zeros((full_freq, pilot_time_idx.size), dtype=np.complex64)
    target_freq = np.arange(full_freq, dtype=np.float32)
    for time_col in range(pilot_time_idx.size):
        known = pilot_values[:, time_col]
        frequency_filled[:, time_col] = _interp_complex_1d(
            pilot_freq_idx.astype(np.float32),
            known,
            target_freq,
        )

    full_grid = np.zeros((full_freq, full_time), dtype=np.complex64)
    target_time = np.arange(full_time, dtype=np.float32)
    for freq_row in range(full_freq):
        known = frequency_filled[freq_row, :]
        full_grid[freq_row, :] = _interp_complex_1d(
            pilot_time_idx.astype(np.float32),
            known,
            target_time,
        )
    return full_grid


def _estimate_channel_ls(
    rx: np.ndarray,
    pilot_freq_idx: np.ndarray,
    pilot_time_idx: np.ndarray,
    known_pilot_symbol: np.complex64,
    estimation_noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    freq = rx.shape[1]
    time = rx.shape[2]
    h_est = np.zeros_like(rx, dtype=np.complex64)
    for ant_idx in range(rx.shape[0]):
        y_grid = rx[ant_idx]
        pilot_ls = y_grid[np.ix_(pilot_freq_idx, pilot_time_idx)] / known_pilot_symbol
        h_est[ant_idx] = _interpolate_pilot_grid(
            pilot_ls,
            pilot_freq_idx,
            pilot_time_idx,
            freq,
            time,
        )

    if estimation_noise_std > 0.0:
        real = rng.normal(scale=estimation_noise_std, size=h_est.shape)
        imag = rng.normal(scale=estimation_noise_std, size=h_est.shape)
        h_est = h_est + (real + 1j * imag).astype(np.complex64)
    return h_est.astype(np.complex64)


def _compose_resource_grid(rx: np.ndarray, ls_estimate: np.ndarray) -> np.ndarray:
    rx_channels = np.concatenate((rx.real, rx.imag), axis=0)
    ls_channels = np.concatenate((ls_estimate.real, ls_estimate.imag), axis=0)
    return np.concatenate((rx_channels, ls_channels), axis=0).astype(np.float32)


def _build_sample(
    channel: np.ndarray,
    cfg: DeepMIMOGenerationConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    freq = cfg.num_subcarriers
    time = cfg.num_ofdm_symbols

    snr_db = np.float32(rng.uniform(cfg.snr_db_min, cfg.snr_db_max))
    snr_linear = 10 ** (float(snr_db) / 10.0)
    noise_std = np.sqrt(0.5 / snr_linear)

    bit_labels = rng.integers(
        0,
        2,
        size=(cfg.bits_per_symbol, freq, time),
        endpoint=False,
        dtype=np.int8,
    ).astype(np.float32)

    pilot_freq_idx = np.arange(0, freq, cfg.pilot_spacing_freq, dtype=np.int64)
    pilot_time_idx = np.asarray(cfg.pilot_spacing_time, dtype=np.int64)
    if pilot_freq_idx.size == 0 or pilot_time_idx.size == 0:
        raise ValueError("Pilot grid is empty. Adjust pilot spacing in config.")
    _stamp_known_pilot_bits(bit_labels, pilot_freq_idx, pilot_time_idx)

    tx_symbols = _modulate(bit_labels)
    channel_3d = np.repeat(channel[:, :, None], time, axis=2)

    noise_real = rng.normal(scale=noise_std, size=channel_3d.shape)
    noise_imag = rng.normal(scale=noise_std, size=channel_3d.shape)
    noise = (noise_real + 1j * noise_imag).astype(np.complex64)
    rx = channel_3d * tx_symbols[None, :, :] + noise

    known_pilot_symbol = np.complex64((-3.0 - 3.0j) / np.sqrt(10.0))
    ls_estimate = _estimate_channel_ls(
        rx,
        pilot_freq_idx,
        pilot_time_idx,
        known_pilot_symbol,
        cfg.estimation_noise_std,
        rng,
    )

    resource_grid = _compose_resource_grid(rx, ls_estimate)
    channel_target = np.concatenate((channel_3d.real, channel_3d.imag), axis=0).astype(np.float32)

    return {
        "inputs": resource_grid,
        "bit_labels": bit_labels,
        "channel_target": channel_target,
        "snr_db": snr_db,
        "channel_profile": cfg.profile_name,
        "modulation": cfg.modulation,
    }


def _extract_deepmimov3_channel_blocks(dataset: Any) -> list[np.ndarray]:
    blocks: list[np.ndarray] = []
    if isinstance(dataset, list):
        for item in dataset:
            blocks.extend(_extract_deepmimov3_channel_blocks(item))
        return blocks
    if isinstance(dataset, dict):
        user_section = dataset.get("user")
        if isinstance(user_section, dict):
            channel = user_section.get("channel")
            if channel is not None:
                blocks.append(np.asarray(channel))
        return blocks
    return blocks


def _infer_deepmimov3_all_user_rows(dataset_folder: str, scenario: str) -> tuple[int, ...] | None:
    try:
        import scipy.io
    except ImportError:  # pragma: no cover - scipy comes with deepmimov3
        return None

    scenario_dir = Path(dataset_folder) / scenario
    candidate_files = [
        scenario_dir / "params.mat",
        scenario_dir / f"{scenario}.params.mat",
    ]
    params_path = next((path for path in candidate_files if path.exists()), None)
    if params_path is None:
        return None

    payload = scipy.io.loadmat(str(params_path))
    user_grids_raw = payload.get("user_grids")
    if user_grids_raw is None:
        return None

    user_grids = np.asarray(user_grids_raw, dtype=np.int64)
    if user_grids.ndim == 1:
        user_grids = user_grids.reshape(1, -1)
    if user_grids.shape[1] < 2:
        return None

    min_row = max(int(user_grids[:, 0].min()) - 1, 0)
    max_row = int(user_grids[:, 1].max()) - 1
    if max_row < min_row:
        return None
    return tuple(range(min_row, max_row + 1))


def _load_channels_from_deepmimo_v3(cfg: DeepMIMOGenerationConfig) -> list[np.ndarray]:
    import DeepMIMOv3 as dm3

    params = dm3.default_params()
    scenario_path = Path(cfg.scenario)
    if scenario_path.is_absolute():
        params["dataset_folder"] = str(scenario_path.parent)
        params["scenario"] = scenario_path.name
    else:
        params["dataset_folder"] = str(cfg.dataset_folder)
        params["scenario"] = cfg.scenario

    params["num_paths"] = int(cfg.num_paths)
    params["active_BS"] = np.asarray(cfg.active_bs, dtype=np.int64)
    if cfg.user_rows is not None:
        params["user_rows"] = np.asarray(cfg.user_rows, dtype=np.int64)
    elif cfg.use_all_rows_if_unspecified:
        all_rows = _infer_deepmimov3_all_user_rows(str(params["dataset_folder"]), str(params["scenario"]))
        if all_rows is not None:
            params["user_rows"] = np.asarray(all_rows, dtype=np.int64)
    params["user_subsampling"] = float(cfg.user_subsampling)
    params["enable_BS2BS"] = 0
    params["OFDM_channels"] = 1
    params["OFDM"]["selected_subcarriers"] = np.arange(cfg.num_subcarriers, dtype=np.int64)
    params["OFDM"]["subcarriers"] = max(int(params["OFDM"]["subcarriers"]), int(cfg.num_subcarriers))
    params["bs_antenna"]["shape"] = np.asarray([1, 1], dtype=np.int64)
    params["ue_antenna"]["shape"] = np.asarray([cfg.num_rx_antennas, 1], dtype=np.int64)

    dataset = dm3.generate_data(params)
    blocks = [_to_four_dim_channel_block(block) for block in _extract_deepmimov3_channel_blocks(dataset)]
    if not blocks:
        raise ValueError("DeepMIMOv3 returned no UE-BS channel blocks.")
    return blocks


def _load_channels_from_deepmimo_v4(cfg: DeepMIMOGenerationConfig) -> list[np.ndarray]:
    import deepmimo as dm

    scenario_path = Path(cfg.scenario)
    if cfg.download_if_missing and not scenario_path.is_absolute():
        dm.download(cfg.scenario)

    dataset = dm.load(cfg.scenario)
    if cfg.active_users_only:
        try:
            dataset = dataset.trim(idxs_mode="active")
        except Exception:
            pass

    raw_channels = dataset.compute_channels()
    blocks = [_to_four_dim_channel_block(block) for block in _flatten_channel_blocks(raw_channels)]
    if not blocks:
        raise ValueError("DeepMIMO returned no channel blocks.")
    return blocks


def _load_channels_from_deepmimo(cfg: DeepMIMOGenerationConfig) -> tuple[np.ndarray, np.ndarray, int, bool]:
    try:
        blocks = _load_channels_from_deepmimo_v3(cfg)
    except ImportError:
        try:
            blocks = _load_channels_from_deepmimo_v4(cfg)
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Missing DeepMIMO dependency. Install DeepMIMOv3 (py3.10) or deepmimo (py3.11+)."
            ) from exc

    if not blocks:
        raise ValueError("DeepMIMO returned no channel blocks.")

    block_sizes = [block.shape[0] for block in blocks]
    total_samples = int(sum(block_sizes))
    sampling_with_replacement = total_samples < cfg.num_samples
    if sampling_with_replacement:
        raise ValueError(
            f"DeepMIMO scenario provides {total_samples} samples, required {cfg.num_samples}. "
            "Adjust generation.num_samples or scenario/trim configuration "
            "(e.g. active_bs/user_rows), or enable replacement sampling."
        )

    select_rng = np.random.default_rng(cfg.seed)
    selected = select_rng.choice(
        total_samples,
        size=cfg.num_samples,
        replace=sampling_with_replacement,
    )
    if not sampling_with_replacement:
        selected = np.sort(selected)
    selected_channels: list[np.ndarray] = []

    cumulative = np.cumsum(block_sizes)
    subcarrier_indices: np.ndarray | None = None
    for global_index in selected.tolist():
        block_index = int(np.searchsorted(cumulative, global_index, side="right"))
        prev_cumulative = int(cumulative[block_index - 1]) if block_index > 0 else 0
        local_index = int(global_index - prev_cumulative)

        user_channel = blocks[block_index][local_index]
        rx_channel = _extract_rx_subcarrier_channel(user_channel, cfg.num_rx_antennas)
        if subcarrier_indices is None:
            subcarrier_indices = _resolve_subcarrier_indices(rx_channel.shape[1], cfg.num_subcarriers)
        rx_channel = rx_channel[:, subcarrier_indices]
        rx_channel = _normalize_channel_power(rx_channel)
        selected_channels.append(rx_channel.astype(np.complex64))

    if subcarrier_indices is None:
        raise ValueError("Failed to resolve DeepMIMO subcarrier indices.")

    return np.stack(selected_channels, axis=0), subcarrier_indices, total_samples, sampling_with_replacement


def generate_deepmimo_arrow_dataset(
    cfg: DeepMIMOGenerationConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Generate a DeepMIMO OOD dataset and save it in Arrow format."""

    channels, subcarrier_indices, unique_channel_candidates, sampled_with_replacement = _load_channels_from_deepmimo(
        cfg
    )
    sample_rng = np.random.default_rng(cfg.seed + 1_000_000)

    def sample_generator():
        for sample_channel in channels:
            yield _build_sample(sample_channel, cfg, sample_rng)

    features = Features(
        {
            "inputs": Array3D(
                shape=(2 * cfg.num_rx_antennas * 2, cfg.num_subcarriers, cfg.num_ofdm_symbols),
                dtype="float32",
            ),
            "bit_labels": Array3D(
                shape=(cfg.bits_per_symbol, cfg.num_subcarriers, cfg.num_ofdm_symbols),
                dtype="float32",
            ),
            "channel_target": Array3D(
                shape=(2 * cfg.num_rx_antennas, cfg.num_subcarriers, cfg.num_ofdm_symbols),
                dtype="float32",
            ),
            "snr_db": Value("float32"),
            "channel_profile": Value("string"),
            "modulation": Value("string"),
        }
    )

    dataset = Dataset.from_generator(sample_generator, features=features)
    if len(dataset) != cfg.num_samples:
        raise ValueError(f"Generated {len(dataset)} samples, expected {cfg.num_samples}")

    target_dir = Path(output_dir)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(target_dir))

    return {
        "channel_profile": cfg.profile_name,
        "modulation": cfg.modulation,
        "num_samples": int(len(dataset)),
        "seed": cfg.seed,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario": cfg.scenario,
        "format": "arrow_dir",
        "subcarrier_indices": subcarrier_indices.tolist(),
        "unique_channel_candidates": unique_channel_candidates,
        "sampled_with_replacement": sampled_with_replacement,
        "snr_db_min": cfg.snr_db_min,
        "snr_db_max": cfg.snr_db_max,
    }
