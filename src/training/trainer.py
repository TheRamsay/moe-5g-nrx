"""Training loop - to be implemented."""

from typing import Any

from src.utils.logging import finish_wandb, setup_wandb


class Trainer:
    """Simple trainer for neural receiver models - to be implemented."""

    def __init__(self, cfg: Any, job_id: str | None = None):
        self.cfg = cfg
        self.job_id = job_id
        self.use_wandb = setup_wandb(cfg, job_id=job_id)
        # TODO: Initialize model, optimizer, loss function
        raise NotImplementedError("Trainer initialization not yet implemented")

    def train_step(self, batch):
        """Single training step - to be implemented."""
        # TODO: Implement forward, backward, optimizer step
        raise NotImplementedError("train_step not yet implemented")

    def train(self, train_loader, num_steps: int):
        """Main training loop - to be implemented."""
        # TODO: Implement training loop with progress bar and logging
        raise NotImplementedError("train not yet implemented")

    def save_checkpoint(self, path: str):
        """Save model checkpoint - to be implemented."""
        # TODO: Implement checkpoint saving
        raise NotImplementedError("save_checkpoint not yet implemented")

    def cleanup(self):
        """Cleanup and finish logging."""
        finish_wandb()
