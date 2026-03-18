"""Simple standard training loop."""

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.logging import finish_wandb, log_metrics, setup_wandb


class Trainer:
    """Simple trainer for neural receiver models."""

    def __init__(self, cfg: Any, job_id: str | None = None):
        self.cfg = cfg
        self.job_id = job_id
        self.device = self._get_device()

        # Setup wandb
        self.use_wandb = setup_wandb(cfg, job_id=job_id)

        # Build model
        self.model = self._build_model().to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
        )

        # Loss function (BCE for bit prediction)
        self.criterion = nn.BCEWithLogitsLoss()

        print(f"[Trainer] Device: {self.device}")
        print(f"[Trainer] Model: {cfg.model.name}")
        print(f"[Trainer] Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _get_device(self) -> torch.device:
        """Get training device."""
        if self.cfg.runtime.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_model(self) -> nn.Module:
        """Build model based on config."""
        from src.models.dense import StaticDenseNRX
        from src.models.moe import MoENRX

        if self.cfg.model.family == "static_dense":
            model = StaticDenseNRX(
                input_channels=self.cfg.data.input_channels,
                hidden_dim=self.cfg.model.backbone.hidden_dim,
                depth=self.cfg.model.backbone.depth,
            )
        elif self.cfg.model.family == "moe":
            experts_dict = {}
            for k, v in self.cfg.model.experts.items():
                experts_dict[k] = {"hidden_dim": v.hidden_dim, "depth": v.depth}
            model = MoENRX(
                input_channels=self.cfg.data.input_channels,
                experts_config=experts_dict,
                temperature=self.cfg.model.router.temperature,
            )
        else:
            raise ValueError(f"Unknown model family: {self.cfg.model.family}")

        return model

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        # Forward
        if self.cfg.model.family == "moe":
            logits, router_weights = self.model(x)

            # Standard loss + compute penalty
            loss = self.criterion(logits, y)

            # Add compute penalty (penalize using heavy experts)
            if self.cfg.model.compute.flops_penalty_alpha > 0 and router_weights is not None:
                # Expected FLOPs based on routing
                expert_costs = torch.tensor([0.25, 0.5, 1.0], device=self.device)
                expected_flops = (router_weights * expert_costs).sum()
                compute_penalty = self.cfg.model.compute.flops_penalty_alpha * expected_flops
                loss = loss + compute_penalty
        else:
            logits = self.model(x)
            loss = self.criterion(logits, y)
            router_weights = None

        # Backward
        loss.backward()
        self.optimizer.step()

        # Metrics
        with torch.no_grad():
            predictions = (torch.sigmoid(logits) > 0.5).float()
            accuracy = (predictions == y).float().mean()
            ber = 1 - accuracy

        metrics: dict[str, float] = {
            "loss": loss.item(),
            "ber": ber.item(),
            "accuracy": accuracy.item(),
        }

        if self.cfg.model.family == "moe" and router_weights is not None:
            metrics["expert_usage_heavy"] = router_weights[:, 2].mean().item()
            metrics["expert_usage_medium"] = router_weights[:, 1].mean().item()
            metrics["expert_usage_tiny"] = router_weights[:, 0].mean().item()

        return metrics

    def train(self, train_loader: DataLoader, num_steps: int) -> None:
        """Main training loop."""
        log_every = self.cfg.logging.log_every_n_steps

        print(f"[Trainer] Starting training for {num_steps} steps")

        step = 0
        epoch = 0

        pbar = tqdm(total=num_steps, desc="Training")

        while step < num_steps:
            for batch in train_loader:
                if step >= num_steps:
                    break

                # Training step
                metrics = self.train_step(batch)

                # Logging
                if step % log_every == 0:
                    pbar.set_postfix({"loss": f"{metrics['loss']:.4f}", "ber": f"{metrics['ber']:.4f}"})

                    if self.use_wandb:
                        log_metrics(metrics, step=step)

                step += 1
                pbar.update(1)

            epoch += 1

        pbar.close()
        print(f"[Trainer] Training complete after {step} steps ({epoch} epochs)")

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.cfg,
            "job_id": self.job_id,
        }
        torch.save(checkpoint, path)
        print(f"[Trainer] Saved checkpoint to {path}")

    def cleanup(self) -> None:
        """Cleanup and finish logging."""
        finish_wandb()
