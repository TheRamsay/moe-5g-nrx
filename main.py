"""Main entry point for training."""

import os
import random
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Prevent TF from using the GPU. TF is pulled in by Sionna but is never used for
# GPU compute when training from cached data. On CC9.0 nodes (H100) TF falls back
# to PTX JIT compilation which blocks the CUDA driver and deadlocks PyTorch's
# pin-memory thread — no training step ever fires.
try:
    import tensorflow as tf  # noqa: E402

    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from src.data import build_train_loader  # noqa: E402
from src.training import Trainer  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the dense baseline training pipeline."""
    _set_seed(int(cfg.runtime.seed))
    print(f"Config: {cfg.model.name} ({cfg.model.family})")

    train_loader = build_train_loader(cfg)
    trainer = Trainer(cfg, job_id=os.environ.get("PBS_JOBID"))

    try:
        trainer.train(train_loader=train_loader, num_steps=int(cfg.training.max_steps))
        checkpoint_path = Path(cfg.training.checkpoint_dir) / f"{cfg.model.name}.pt"
        trainer.save_checkpoint(
            str(checkpoint_path),
            artifact_aliases=["latest", "final", f"step-{trainer.global_step}"],
            artifact_primary_alias="latest",
            log_artifact=True,
            checkpoint_kind="final",
        )
        print(f"Saved checkpoint to {checkpoint_path}")
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
