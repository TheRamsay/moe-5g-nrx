"""Warm-start utilities: load dense checkpoints into MoE expert branches."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _load_state_dict(source: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Return model_state_dict from a local path or W&B artifact reference.

    W&B artifact format: ``entity/project/artifact_name:alias``
    Local path: any existing ``.pt`` file.
    """
    path = Path(source)
    if not path.exists():
        import wandb  # lazy import — not needed for non-warm-start runs

        api = wandb.Api()
        artifact = api.artifact(source)
        artifact_dir = Path(artifact.download())
        pt_files = sorted(artifact_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt file found in downloaded artifact {source!r}")
        path = pt_files[0]
        logger.info("Downloaded artifact %s -> %s", source, path)

    ckpt: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    return ckpt["model_state_dict"]


def load_warm_start(
    moe_model: Any,
    *,
    stem_checkpoint: str,
    expert_checkpoints: dict[str, str],
    device: str = "cpu",
) -> None:
    """Inject pretrained dense weights into a MoENRX model in-place.

    ``stem_checkpoint``    — local path or W&B artifact ref for the stem
                             (use the large dense checkpoint; stem dims must match)
    ``expert_checkpoints`` — mapping of expert_name -> path or W&B artifact ref

    Dense-to-MoE weight mapping
    ---------------------------
    Dense ``StaticDenseNRX`` attributes -> ``_ExpertHead`` attributes:
        stem.*             -> moe_model.stem.*
        backbone.*         -> moe_model.experts[name].backbone.*
        readout_llrs.*     -> moe_model.experts[name].readout_llrs.*
        readout_channel.*  -> moe_model.experts[name].readout_channel.*
    """
    # --- stem ---
    state = _load_state_dict(stem_checkpoint, device=device)
    stem_weights = {k[len("stem.") :]: v for k, v in state.items() if k.startswith("stem.")}
    missing, unexpected = moe_model.stem.load_state_dict(stem_weights, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Stem weight mismatch — missing={missing}, unexpected={unexpected}")
    print(f"[warm-start] stem loaded from {stem_checkpoint}")

    # --- experts ---
    for name, source in expert_checkpoints.items():
        if name not in moe_model.experts:
            raise ValueError(f"Expert {name!r} not found in model. " f"Available: {list(moe_model.experts.keys())}")
        state = _load_state_dict(source, device=device)
        expert_weights = {
            k: v for k, v in state.items() if k.startswith(("backbone.", "readout_llrs.", "readout_channel."))
        }
        missing, unexpected = moe_model.experts[name].load_state_dict(expert_weights, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Expert {name!r} weight mismatch — missing={missing}, unexpected={unexpected}")
        print(f"[warm-start] expert {name!r} loaded from {source}")

    print(f"[warm-start] complete — stem + experts ({', '.join(expert_checkpoints)}) ready")
