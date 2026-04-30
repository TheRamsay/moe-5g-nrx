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
        # Function-specialized experts may have a subset of {backbone, readout_llrs,
        # readout_channel} — e.g. channel_only has no readout_llrs, sink has nothing.
        # Filter the dense checkpoint to only keys present in the target expert,
        # then load with strict=False so warm-start works across architecture variants.
        state = _load_state_dict(source, device=device)
        expert_weights = {
            k: v for k, v in state.items() if k.startswith(("backbone.", "readout_llrs.", "readout_channel."))
        }
        target_keys = set(moe_model.experts[name].state_dict().keys())
        if target_keys:
            expert_weights = {k: v for k, v in expert_weights.items() if k in target_keys}
        if not expert_weights:
            print(f"[warm-start] expert {name!r}: no matching keys in checkpoint, skipping")
            continue
        missing, unexpected = moe_model.experts[name].load_state_dict(expert_weights, strict=False)
        if unexpected:
            raise RuntimeError(f"Expert {name!r} unexpected keys after filter: {unexpected}")
        skipped = [k for k in target_keys if k not in expert_weights]
        suffix = f" (skipped {len(skipped)} non-matching keys)" if skipped else ""
        print(f"[warm-start] expert {name!r} loaded from {source}{suffix}")

    print(f"[warm-start] complete — stem + experts ({', '.join(expert_checkpoints)}) ready")
