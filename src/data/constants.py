from __future__ import annotations

from enum import Enum
from typing import Final


class ChannelProfile(str, Enum):
    """Supported synthetic channel models for training/evaluation."""

    UMA = "uma"
    TDLC = "tdlc"
    TDLA = "tdla"  # 3GPP TDL-A: low delay spread NLOS
    TDLD = "tdld"  # 3GPP TDL-D: low delay spread Rician (LOS-dominant)
    CDLA = "cdla"  # 3GPP CDL-A: clustered delay line, NLOS with spatial structure
    MIXED = "mixed"  # Equal sampling from UMA and TDL-C


class Modulation(str, Enum):
    """Enumerates supported constellations."""

    QAM16 = "qam16"


BITS_PER_SYMBOL: Final[dict[Modulation, int]] = {
    Modulation.QAM16: 4,
}

# Seed namespace offsets used to avoid train/val/test overlap.
TRAIN_ITERATION_SEED_OFFSET: Final[int] = 1_000_000
TRAIN_WORKER_SEED_OFFSET: Final[int] = 100_000
MIXED_SIMULATOR_SEED_OFFSET: Final[int] = 50_000
TRAIN_DATASET_SEED_OFFSET: Final[int] = 200_000_000
VAL_SEED_OFFSET: Final[int] = 500_000_000
TEST_SEED_OFFSET: Final[int] = 1_000_000_000
MIXED_PROFILE_SEED_OFFSET: Final[int] = 1_000_000

# Profiles that can be used for data generation (excludes MIXED which is composed)
GENERATABLE_PROFILES: Final[list[str]] = ["uma", "tdlc", "tdla", "tdld", "cdla"]


def resolve_modulation(name: str | Modulation) -> Modulation:
    """Normalize user-provided modulation strings."""

    if isinstance(name, Modulation):
        return name
    normalized = name.lower()
    try:
        return Modulation(normalized)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported modulation: {name}") from exc


def resolve_channel_profile(name: str | ChannelProfile) -> ChannelProfile:
    """Normalize channel profile names."""

    if isinstance(name, ChannelProfile):
        return name
    normalized = name.lower()
    try:
        return ChannelProfile(normalized)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported channel profile: {name}") from exc


def bits_per_symbol(modulation: str | Modulation) -> int:
    """Return the number of coded bits per symbol for the modulation."""

    modulation_enum = resolve_modulation(modulation)
    return BITS_PER_SYMBOL[modulation_enum]
