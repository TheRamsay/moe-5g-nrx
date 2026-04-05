"""MoE 5G Neural Receiver package."""

from src.models import MoENRX, StaticDenseNRX

__all__ = [
    "StaticDenseNRX",
    "MoENRX",
    "Trainer",
]


def __getattr__(name: str):
    if name == "Trainer":
        from src.training import Trainer

        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
