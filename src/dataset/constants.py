from __future__ import annotations

from enum import Enum
from typing import Final


class ChannelProfile(str, Enum):
    """Supported synthetic channel models."""

    AWGN = "awgn"
    UMA = "uma"
    TDLC = "tdlc"


class Modulation(str, Enum):
    """Enumerates supported constellations."""

    QAM16 = "qam16"


BITS_PER_SYMBOL: Final[dict[Modulation, int]] = {
    Modulation.QAM16: 4,
}


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
