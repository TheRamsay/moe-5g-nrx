from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import sionna
import tensorflow as tf
from sionna.channel.tr38901 import CDL, TDL, AntennaArray, UMa

from .constants import (
    ChannelProfile,
    Modulation,
    bits_per_symbol,
    resolve_channel_profile,
    resolve_modulation,
)


@dataclass(frozen=True)
class SimulatorConfig:
    """Configuration for the synthetic Sionna-compatible simulator."""

    num_rx_antennas: int
    num_subcarriers: int
    num_ofdm_symbols: int
    channel_profile: ChannelProfile
    modulation: Modulation
    snr_db_min: float
    snr_db_max: float
    estimation_noise_std: float
    carrier_frequency_hz: float
    subcarrier_spacing_hz: float
    delay_spread_s: float
    min_speed_mps: float
    max_speed_mps: float
    pilot_spacing_freq: int
    pilot_spacing_time: list[int]
    uma_cell_radius_m: float
    bs_height_m: float
    ut_height_m: float
    indoor_probability: float
    o2i_model: str
    seed: int | None = None


@dataclass(frozen=True)
class SimulatorBatch:
    """Container holding a generated mini-batch."""

    resource_grid: np.ndarray  # (batch, channels, freq, time)
    bit_labels: np.ndarray  # (batch, bits_per_symbol, freq, time)
    channel_target: np.ndarray  # (batch, 2*num_rx_antennas, freq, time)
    snr_db: np.ndarray  # (batch,)
    metadata: dict[str, Any]


class SionnaNRXSimulator:
    """Sionna-backed simulator producing compute-aware NRX training data."""

    def __init__(self, config: SimulatorConfig) -> None:
        self.cfg = config
        self._bits_per_symbol = bits_per_symbol(config.modulation)
        self._seed_base = config.seed
        self._batch_index = 0
        self._rng = np.random.default_rng(config.seed)
        self._tf_rng = self._build_tf_rng(config.seed)
        self._tdl_channels: dict[str, TDL] = {}
        self._cdl_channels: dict[str, CDL] = {}
        self._uma_channel: UMa | None = None
        if config.seed is not None:
            sionna.config.seed = int(config.seed)

    def reseed(self, seed: int | None) -> None:
        """Reset both NumPy and TensorFlow RNG streams."""

        self._seed_base = seed
        self._batch_index = 0
        self._rng = np.random.default_rng(seed)
        self._tf_rng = self._build_tf_rng(seed)
        if seed is not None:
            sionna.config.seed = int(seed)

    def _next_batch_seed(self) -> int | None:
        if self._seed_base is None:
            return None
        bounded_seed = int(self._seed_base + self._batch_index) % (2**31 - 1)
        self._batch_index += 1
        return bounded_seed

    def _build_tf_rng(self, seed: int | None) -> tf.random.Generator:
        if seed is None:
            return tf.random.Generator.from_non_deterministic_state()
        bounded_seed = int(seed) % (2**31 - 1)
        return tf.random.Generator.from_seed(bounded_seed)

    def generate_batch(self, batch_size: int) -> SimulatorBatch:
        """Generate a mini-batch of resource grids and labels."""

        batch_seed = self._next_batch_seed()
        if batch_seed is not None:
            tf.random.set_seed(batch_seed)
            sionna.config.seed = int(batch_seed)

        freq = self.cfg.num_subcarriers
        time = self.cfg.num_ofdm_symbols
        snr_db = self._rng.uniform(
            self.cfg.snr_db_min,
            self.cfg.snr_db_max,
            size=(batch_size,),
        ).astype(np.float32)
        bit_labels = self._rng.integers(
            0,
            2,
            size=(batch_size, self._bits_per_symbol, freq, time),
            endpoint=False,
            dtype=np.int8,
        ).astype(np.float32)
        pilot_freq_idx = np.arange(0, freq, self.cfg.pilot_spacing_freq, dtype=np.int64)
        pilot_time_idx = np.array(self.cfg.pilot_spacing_time, dtype=np.int64)
        if pilot_freq_idx.size == 0 or pilot_time_idx.size == 0:
            raise ValueError("Pilot grid is empty. Adjust pilot spacing in dataset config.")
        self._stamp_known_pilot_bits(bit_labels, pilot_freq_idx, pilot_time_idx)
        tx_symbols = self._modulate(bit_labels)
        channel = self._sample_channel(batch_size, freq, time)
        noise = self._sample_noise(batch_size, freq, time, snr_db)
        rx = channel * tx_symbols[:, None, :, :] + noise
        known_pilot_symbol = self._known_pilot_symbol()
        ls_estimate = self._estimate_channel_ls(
            rx=rx,
            pilot_freq_idx=pilot_freq_idx,
            pilot_time_idx=pilot_time_idx,
            known_pilot_symbol=known_pilot_symbol,
        )
        resource_grid = self._compose_resource_grid(rx, ls_estimate)
        channel_target = np.concatenate((channel.real, channel.imag), axis=1)
        batch_metadata = {
            "channel_profile": self.cfg.channel_profile.value,
            "modulation": self.cfg.modulation.value,
            "pilot_spacing_freq": self.cfg.pilot_spacing_freq,
            "pilot_spacing_time": self.cfg.pilot_spacing_time,
        }
        return SimulatorBatch(
            resource_grid=resource_grid.astype(np.float32),
            bit_labels=bit_labels.astype(np.float32),
            channel_target=channel_target.astype(np.float32),
            snr_db=snr_db,
            metadata=batch_metadata,
        )

    def _compose_resource_grid(self, rx: np.ndarray, ls_estimate: np.ndarray) -> np.ndarray:
        rx_real = rx.real
        rx_imag = rx.imag
        ls_real = ls_estimate.real
        ls_imag = ls_estimate.imag
        rx_channels = np.concatenate((rx_real, rx_imag), axis=1)
        ls_channels = np.concatenate((ls_real, ls_imag), axis=1)
        return np.concatenate((rx_channels, ls_channels), axis=1)

    def _modulate(self, bit_labels: np.ndarray) -> np.ndarray:
        if self.cfg.modulation is not Modulation.QAM16:
            raise NotImplementedError("Only 16-QAM modulation is implemented.")
        inphase_bits = bit_labels[:, 0:2, :, :]
        quadrature_bits = bit_labels[:, 2:4, :, :]
        inphase_level = self._pam4_from_bits(inphase_bits)
        quadrature_level = self._pam4_from_bits(quadrature_bits)
        normalization = np.sqrt(10.0)
        return (inphase_level + 1j * quadrature_level) / normalization

    def _pam4_from_bits(self, bits: np.ndarray) -> np.ndarray:
        bit0 = bits[:, 0, :, :]
        bit1 = bits[:, 1, :, :]
        indices = bit0 * 2 + bit1
        mapping = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
        return mapping[indices.astype(np.int8)]

    def _known_pilot_symbol(self) -> np.complex64:
        # 16-QAM symbol for 0000 under this mapper: (-3 - 3j) / sqrt(10)
        return np.complex64((-3.0 - 3.0j) / np.sqrt(10.0))  # TODO: correct??

    def _stamp_known_pilot_bits(
        self,
        bit_labels: np.ndarray,
        pilot_freq_idx: np.ndarray,
        pilot_time_idx: np.ndarray,
    ) -> None:
        for freq_idx in pilot_freq_idx:
            for time_idx in pilot_time_idx:
                bit_labels[:, :, freq_idx, time_idx] = 0.0

    def _sample_channel(self, batch_size: int, freq: int, time: int) -> np.ndarray:
        if self.cfg.channel_profile is ChannelProfile.UMA:
            return self._sample_uma_channel(batch_size, freq, time)

        tdl_model = {
            ChannelProfile.TDLA: "A",
            ChannelProfile.TDLC: "C",
            ChannelProfile.TDLD: "D",
        }.get(self.cfg.channel_profile)
        if tdl_model is not None:
            channel = self._get_or_create_tdl(tdl_model)
            sampling_frequency = self.cfg.subcarrier_spacing_hz * float(freq)
            cir_coeffs, cir_delays = channel(
                batch_size=batch_size,
                num_time_steps=time,
                sampling_frequency=sampling_frequency,
            )
            response = self._cir_to_ofdm_response(cir_coeffs, cir_delays, freq)
            return self._normalize_channel_power(response)

        cdl_model = {
            ChannelProfile.CDLA: "A",
        }.get(self.cfg.channel_profile)
        if cdl_model is not None:
            channel = self._get_or_create_cdl(cdl_model)
            sampling_frequency = self.cfg.subcarrier_spacing_hz * float(freq)
            cir_coeffs, cir_delays = channel(
                batch_size=batch_size,
                num_time_steps=time,
                sampling_frequency=sampling_frequency,
            )
            response = self._cir_to_ofdm_response(cir_coeffs, cir_delays, freq)
            return self._normalize_channel_power(response)

        raise ValueError(f"Unsupported channel profile: {self.cfg.channel_profile}")

    def _sample_uma_channel(self, batch_size: int, freq: int, time: int) -> np.ndarray:
        channel = self._get_or_create_uma_channel()
        ut_loc, bs_loc = self._sample_uma_positions(batch_size)
        ut_orientations = tf.zeros([batch_size, 1, 3], dtype=tf.float32)
        bs_orientations = tf.zeros([batch_size, 1, 3], dtype=tf.float32)
        ut_velocities = self._sample_uma_velocities(batch_size)
        in_state = self._sample_indoor_state(batch_size)

        channel.set_topology(
            ut_loc=ut_loc,
            bs_loc=bs_loc,
            ut_orientations=ut_orientations,
            bs_orientations=bs_orientations,
            ut_velocities=ut_velocities,
            in_state=in_state,
            los=None,
        )

        sampling_frequency = self.cfg.subcarrier_spacing_hz * float(freq)
        cir_coeffs, cir_delays = channel(
            num_time_samples=time,
            sampling_frequency=sampling_frequency,
        )
        response = self._cir_to_ofdm_response(cir_coeffs, cir_delays, freq)
        return self._normalize_channel_power(response)

    def _get_or_create_uma_channel(self) -> UMa:
        if self._uma_channel is not None:
            return self._uma_channel

        ut_array = AntennaArray(
            num_rows=1,
            num_cols=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=self.cfg.carrier_frequency_hz,
            dtype=tf.complex64,
        )
        bs_array = AntennaArray(
            num_rows=1,
            num_cols=self.cfg.num_rx_antennas,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=self.cfg.carrier_frequency_hz,
            dtype=tf.complex64,
        )
        self._uma_channel = UMa(
            carrier_frequency=self.cfg.carrier_frequency_hz,
            o2i_model=self.cfg.o2i_model,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            dtype=tf.complex64,
        )
        return self._uma_channel

    def _sample_uma_positions(self, batch_size: int) -> tuple[tf.Tensor, tf.Tensor]:
        radius = self.cfg.uma_cell_radius_m
        angles = self._tf_rng.uniform([batch_size, 1], minval=0.0, maxval=2.0 * np.pi, dtype=tf.float32)
        radial = tf.sqrt(self._tf_rng.uniform([batch_size, 1], minval=0.0, maxval=1.0, dtype=tf.float32)) * radius

        ut_x = radial * tf.cos(angles)
        ut_y = radial * tf.sin(angles)
        ut_z = tf.fill([batch_size, 1], tf.cast(self.cfg.ut_height_m, tf.float32))
        ut_loc = tf.stack([ut_x, ut_y, ut_z], axis=-1)

        bs_x = tf.zeros([batch_size, 1], dtype=tf.float32)
        bs_y = tf.zeros([batch_size, 1], dtype=tf.float32)
        bs_z = tf.fill([batch_size, 1], tf.cast(self.cfg.bs_height_m, tf.float32))
        bs_loc = tf.stack([bs_x, bs_y, bs_z], axis=-1)
        return ut_loc, bs_loc

    def _sample_uma_velocities(self, batch_size: int) -> tf.Tensor:
        speed = self._tf_rng.uniform(
            [batch_size, 1],
            minval=float(self.cfg.min_speed_mps),
            maxval=float(self.cfg.max_speed_mps),
            dtype=tf.float32,
        )
        heading = self._tf_rng.uniform([batch_size, 1], minval=0.0, maxval=2.0 * np.pi, dtype=tf.float32)
        vx = speed * tf.cos(heading)
        vy = speed * tf.sin(heading)
        vz = tf.zeros_like(vx)
        return tf.stack([vx, vy, vz], axis=-1)

    def _sample_indoor_state(self, batch_size: int) -> tf.Tensor:
        indoor_prob = tf.cast(self.cfg.indoor_probability, tf.float32)
        random_draw = self._tf_rng.uniform([batch_size, 1], minval=0.0, maxval=1.0, dtype=tf.float32)
        return random_draw < indoor_prob

    def _get_or_create_tdl(self, model: str) -> TDL:
        cached = self._tdl_channels.get(model)
        if cached is not None:
            return cached

        tdl = TDL(
            model=model,
            delay_spread=self.cfg.delay_spread_s,
            carrier_frequency=self.cfg.carrier_frequency_hz,
            min_speed=self.cfg.min_speed_mps,
            max_speed=self.cfg.max_speed_mps,
            num_rx_ant=self.cfg.num_rx_antennas,
            num_tx_ant=1,
            dtype=tf.complex64,
        )
        self._tdl_channels[model] = tdl
        return tdl

    def _get_or_create_cdl(self, model: str) -> CDL:
        cached = self._cdl_channels.get(model)
        if cached is not None:
            return cached

        # CDL needs antenna arrays (unlike TDL). Same SIMO 1x4 layout as UMa.
        ut_array = AntennaArray(
            num_rows=1,
            num_cols=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=self.cfg.carrier_frequency_hz,
            dtype=tf.complex64,
        )
        bs_array = AntennaArray(
            num_rows=1,
            num_cols=self.cfg.num_rx_antennas,
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=self.cfg.carrier_frequency_hz,
            dtype=tf.complex64,
        )
        cdl = CDL(
            model=model,
            delay_spread=self.cfg.delay_spread_s,
            carrier_frequency=self.cfg.carrier_frequency_hz,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            min_speed=self.cfg.min_speed_mps,
            max_speed=self.cfg.max_speed_mps,
            dtype=tf.complex64,
        )
        self._cdl_channels[model] = cdl
        return cdl

    def _cir_to_ofdm_response(
        self,
        cir_coeffs: tf.Tensor,
        cir_delays: tf.Tensor,
        num_subcarriers: int,
    ) -> np.ndarray:
        frequencies_hz = (
            tf.cast(tf.range(num_subcarriers), tf.float32) - tf.cast(num_subcarriers // 2, tf.float32)
        ) * tf.cast(self.cfg.subcarrier_spacing_hz, tf.float32)
        frequencies_hz = tf.reshape(frequencies_hz, [1, 1, 1, 1, 1, 1, num_subcarriers, 1])

        batch_size = tf.shape(cir_delays)[0]
        num_paths = tf.shape(cir_delays)[-1]
        path_delays = tf.cast(cir_delays, tf.float32)
        path_delays = tf.reshape(path_delays, [batch_size, 1, 1, 1, 1, num_paths, 1, 1])

        phase = tf.exp(
            tf.complex(
                tf.zeros_like(path_delays * frequencies_hz),
                -2.0 * np.pi * path_delays * frequencies_hz,
            )
        )
        coeffs_expanded = tf.expand_dims(cir_coeffs, axis=6)
        frequency_response = tf.reduce_sum(coeffs_expanded * phase, axis=5)
        frequency_response = tf.squeeze(frequency_response, axis=[1, 3, 4])
        return frequency_response.numpy().astype(np.complex64)

    def _normalize_channel_power(self, channel: np.ndarray) -> np.ndarray:
        """Normalize each sample to unit average channel power.

        This keeps configured SNR ranges comparable across channel models. Without
        normalization, UMa includes large-scale path loss while TDL-C is near unit
        power, which makes the same nominal SNR correspond to drastically different
        effective receive conditions.
        """

        power = np.mean(np.abs(channel) ** 2, axis=(1, 2, 3), keepdims=True)
        power = np.maximum(power, 1e-12)
        return channel / np.sqrt(power)

    def _sample_noise(self, batch_size: int, freq: int, time: int, snr_db: np.ndarray) -> np.ndarray:
        shape = (batch_size, self.cfg.num_rx_antennas, freq, time)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_std = np.sqrt(0.5 / snr_linear).astype(np.float32).reshape(batch_size, 1, 1, 1)
        noise_std_tf = tf.convert_to_tensor(noise_std, dtype=tf.float32)
        real = self._tf_rng.normal(shape=shape, dtype=tf.float32) * noise_std_tf
        imag = self._tf_rng.normal(shape=shape, dtype=tf.float32) * noise_std_tf
        noise = tf.complex(real, imag)
        return noise.numpy().astype(np.complex64)

    def _estimate_channel_ls(
        self,
        rx: np.ndarray,
        pilot_freq_idx: np.ndarray,
        pilot_time_idx: np.ndarray,
        known_pilot_symbol: np.complex64,
    ) -> np.ndarray:
        freq = rx.shape[2]
        time = rx.shape[3]
        if np.abs(known_pilot_symbol) < 1e-6:
            raise ValueError("Known pilot symbol magnitude must be non-zero.")
        h_est = np.zeros_like(rx, dtype=np.complex64)
        for batch_idx in range(rx.shape[0]):
            for ant_idx in range(rx.shape[1]):
                y_grid = rx[batch_idx, ant_idx]
                pilot_ls = y_grid[np.ix_(pilot_freq_idx, pilot_time_idx)] / known_pilot_symbol
                h_est[batch_idx, ant_idx] = self._interpolate_pilot_grid(
                    pilot_ls,
                    pilot_freq_idx,
                    pilot_time_idx,
                    freq,
                    time,
                )

        if self.cfg.estimation_noise_std > 0.0:
            h_est = h_est + self._complex_gaussian(
                h_est.shape,
                scale=float(self.cfg.estimation_noise_std),
            )
        return h_est.astype(np.complex64)

    def _interpolate_pilot_grid(
        self,
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
            frequency_filled[:, time_col] = self._interp_complex_1d(
                pilot_freq_idx.astype(np.float32),
                known,
                target_freq,
            )

        full_grid = np.zeros((full_freq, full_time), dtype=np.complex64)
        target_time = np.arange(full_time, dtype=np.float32)
        for freq_row in range(full_freq):
            known = frequency_filled[freq_row, :]
            full_grid[freq_row, :] = self._interp_complex_1d(
                pilot_time_idx.astype(np.float32),
                known,
                target_time,
            )
        return full_grid

    def _interp_complex_1d(
        self,
        known_x: np.ndarray,
        known_y: np.ndarray,
        target_x: np.ndarray,
    ) -> np.ndarray:
        if known_x.size == 1:
            return np.full(target_x.shape, known_y[0], dtype=np.complex64)
        interp_real = np.interp(target_x, known_x, known_y.real)
        interp_imag = np.interp(target_x, known_x, known_y.imag)
        return (interp_real + 1j * interp_imag).astype(np.complex64)

    def _complex_gaussian(self, shape: tuple[int, ...], scale: float | np.ndarray) -> np.ndarray:
        real = self._rng.normal(scale=scale, size=shape)
        imag = self._rng.normal(scale=scale, size=shape)
        return (real + 1j * imag).astype(np.complex64)


def build_simulator_from_cfg(cfg: Any) -> SionnaNRXSimulator:
    """Construct a simulator instance from an OmegaConf-like object."""

    channel_profile = resolve_channel_profile(cfg.dataset.channel_profile)
    modulation = resolve_modulation(cfg.dataset.modulation)
    sim_config = SimulatorConfig(
        num_rx_antennas=cfg.data.num_rx_antennas,
        num_subcarriers=cfg.dataset.num_subcarriers,
        num_ofdm_symbols=cfg.dataset.num_ofdm_symbols,
        channel_profile=channel_profile,
        modulation=modulation,
        snr_db_min=cfg.dataset.snr_db.min,
        snr_db_max=cfg.dataset.snr_db.max,
        estimation_noise_std=cfg.dataset.estimation_noise_std,
        carrier_frequency_hz=float(getattr(cfg.dataset, "carrier_frequency_hz", 3.5e9)),
        subcarrier_spacing_hz=float(getattr(cfg.dataset, "subcarrier_spacing_hz", 15e3)),
        delay_spread_s=float(getattr(cfg.dataset, "delay_spread_s", 100e-9)),
        min_speed_mps=float(getattr(cfg.dataset, "min_speed_mps", 0.0)),
        max_speed_mps=float(getattr(cfg.dataset, "max_speed_mps", 30.0)),
        pilot_spacing_freq=int(getattr(cfg.dataset, "pilot_spacing_freq", 2)),
        pilot_spacing_time=list(getattr(cfg.dataset, "pilot_spacing_time", [2, 11])),
        uma_cell_radius_m=float(getattr(cfg.dataset, "uma_cell_radius_m", 500.0)),
        bs_height_m=float(getattr(cfg.dataset, "bs_height_m", 25.0)),
        ut_height_m=float(getattr(cfg.dataset, "ut_height_m", 1.5)),
        indoor_probability=float(getattr(cfg.dataset, "indoor_probability", 0.2)),
        o2i_model=str(getattr(cfg.dataset, "o2i_model", "low")),
        seed=cfg.runtime.seed,
    )
    return SionnaNRXSimulator(sim_config)
