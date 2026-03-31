"""Training loop for dense neural receiver baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import CachedNRXBatch, build_cached_dataloader
from src.models import StaticDenseNRX
from src.utils.logging import finish_wandb, log_metrics, setup_wandb


class Trainer:
    """Simple trainer for neural receiver models."""

    def __init__(self, cfg: Any, job_id: str | None = None):
        self.cfg = cfg
        self.job_id = job_id
        self.device = self._resolve_device(str(cfg.runtime.device))
        self.use_wandb = setup_wandb(cfg, job_id=job_id)
        self.model = self._build_model().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.training.learning_rate),
            weight_decay=float(cfg.training.weight_decay),
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.channel_loss_fn = torch.nn.MSELoss()
        self.global_step = 0
        self._val_dataloader: DataLoader | None = None
        self._setup_validation()

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, Any]:
        """Run a single optimization step."""
        self.model.train()
        features, targets, channel_targets = batch
        features = features.to(self.device)
        targets = targets.to(self.device)
        channel_targets = channel_targets.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(features)
        logits = outputs["logits"]
        channel_estimate = outputs["channel_estimate"]
        loss = self._compute_loss(outputs, targets, channel_targets)
        loss.backward()

        clip_value = float(self.cfg.training.gradient_clip_norm)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value))
        self.optimizer.step()

        predictions = (logits > 0.0).to(dtype=targets.dtype)
        bit_errors = (predictions != targets).to(dtype=torch.float32)
        bit_error_rate = bit_errors.mean()
        block_error_rate = bit_errors.any(dim=1).to(dtype=torch.float32).mean()
        bit_errors_per_block = bit_errors.sum(dim=1)

        bits_per_symbol = int(self.cfg.data.bits_per_symbol)
        symbol_errors = bit_errors.view(bit_errors.shape[0], -1, bits_per_symbol).any(dim=-1).to(dtype=torch.float32)
        symbol_error_rate = symbol_errors.mean()

        block_error_quantiles = torch.quantile(bit_errors_per_block, torch.tensor([0.5, 0.9], device=self.device))
        channel_mse = self.channel_loss_fn(channel_estimate, channel_targets)

        return {
            "loss": float(loss.detach().cpu().item()),
            "ber": float(bit_error_rate.detach().cpu().item()),
            "bler": float(block_error_rate.detach().cpu().item()),
            "ser": float(symbol_error_rate.detach().cpu().item()),
            "block_bit_errors_mean": float(bit_errors_per_block.mean().detach().cpu().item()),
            "block_bit_errors_median": float(block_error_quantiles[0].detach().cpu().item()),
            "block_bit_errors_p90": float(block_error_quantiles[1].detach().cpu().item()),
            "block_bit_errors_max": float(bit_errors_per_block.max().detach().cpu().item()),
            "channel_mse": float(channel_mse.detach().cpu().item()),
            "grad_norm": grad_norm,
            "block_bit_errors_hist": wandb.Histogram(bit_errors_per_block.detach().cpu().numpy()),
        }

    def train(self, train_loader, num_steps: int) -> None:
        """Main training loop with periodic metric logging and validation."""
        data_iterator = iter(train_loader)
        progress = tqdm(total=num_steps, desc="train", dynamic_ncols=True)

        val_enabled = self._validation_enabled()
        val_every_n = int(self.cfg.validation.every_n_steps) if val_enabled else 0

        while self.global_step < num_steps:
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_loader)
                batch = next(data_iterator)

            metrics = self.train_step(batch)
            self.global_step += 1

            progress.update(1)
            progress.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                ber=f"{metrics['ber']:.4f}",
                bler=f"{metrics['bler']:.4f}",
                ser=f"{metrics['ser']:.4f}",
            )

            metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
            if self.use_wandb:
                log_metrics(self._format_logging_metrics(metrics), step=self.global_step)
            else:
                metrics.pop("block_bit_errors_hist", None)

            if self.global_step == 1 or self.global_step % int(self.cfg.logging.log_every_n_steps) == 0:
                print(
                    f"step={self.global_step} loss={metrics['loss']:.4f} "
                    f"ber={metrics['ber']:.4f} ser={metrics['ser']:.4f} bler={metrics['bler']:.4f} "
                    f"block_bits(mean/p90/max)="
                    f"{metrics['block_bit_errors_mean']:.1f}/{metrics['block_bit_errors_p90']:.1f}/{metrics['block_bit_errors_max']:.1f}"
                )

            # Periodic validation
            if val_enabled and self.global_step % val_every_n == 0:
                val_metrics = self.validate()
                if self.use_wandb:
                    log_metrics(self._format_val_metrics(val_metrics), step=self.global_step)
                print(
                    f"  [VAL] step={self.global_step} loss={val_metrics['loss']:.4f} "
                    f"ber={val_metrics['ber']:.4f} bler={val_metrics['bler']:.4f}"
                )

        progress.close()

    @staticmethod
    def _format_logging_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        """Namespace training metrics for cleaner WandB plots."""

        return {f"train/{key}": value for key, value in metrics.items()}

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint and resolved config."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "config": OmegaConf.to_container(self.cfg, resolve=True),
            },
            checkpoint_path,
        )

    def cleanup(self):
        """Cleanup and finish logging."""
        finish_wandb()

    def _build_model(self) -> torch.nn.Module:
        if str(self.cfg.model.family) != "static_dense":
            raise NotImplementedError(f"Model family '{self.cfg.model.family}' is not implemented yet")

        return StaticDenseNRX(
            input_channels=int(self.cfg.data.input_channels),
            output_bits=int(self.cfg.data.bits_per_symbol)
            * int(self.cfg.data.num_subcarriers)
            * int(self.cfg.data.num_ofdm_symbols),
            state_dim=int(self.cfg.model.backbone.state_dim),
            num_cnn_blocks=int(self.cfg.model.backbone.num_cnn_blocks),
            stem_hidden_dims=list(self.cfg.model.backbone.stem_hidden_dims),
            block_hidden_dim=int(self.cfg.model.backbone.block_hidden_dim),
            readout_hidden_dim=int(self.cfg.model.backbone.readout_hidden_dim),
            activation=str(self.cfg.model.backbone.activation),
            num_subcarriers=int(self.cfg.data.num_subcarriers),
            num_ofdm_symbols=int(self.cfg.data.num_ofdm_symbols),
            bits_per_symbol=int(self.cfg.data.bits_per_symbol),
            num_rx_antennas=int(self.cfg.data.num_rx_antennas),
            pilot_symbol_indices=list(self.cfg.model.backbone.pilot_symbol_indices),
            pilot_subcarrier_spacing=int(self.cfg.model.backbone.pilot_subcarrier_spacing),
        )

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            print("[WARNING] CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device(device_name)

    def _compute_loss(
        self,
        outputs: dict[str, torch.Tensor | list[torch.Tensor]],
        targets: torch.Tensor,
        channel_targets: torch.Tensor,
    ) -> torch.Tensor:
        channel_weight = float(self.cfg.model.compute.channel_loss_weight)
        final_loss = self.loss_fn(outputs["logits"], targets) + channel_weight * self.channel_loss_fn(
            outputs["channel_estimate"], channel_targets
        )

        intermediate_logits = outputs["intermediate_logits"]
        intermediate_channel_estimates = outputs["intermediate_channel_estimates"]
        if not intermediate_logits or not intermediate_channel_estimates:
            return final_loss

        multi_loss = torch.zeros((), device=self.device)
        for logits, channel_estimate in zip(intermediate_logits, intermediate_channel_estimates):
            multi_loss = multi_loss + self.loss_fn(logits, targets)
            multi_loss = multi_loss + channel_weight * self.channel_loss_fn(channel_estimate, channel_targets)
        return multi_loss

    def _setup_validation(self) -> None:
        """Initialize validation dataloader if enabled."""
        if not self._validation_enabled():
            return

        val_path = Path(self.cfg.validation.dataset_path)
        if not val_path.exists():
            print(f"[WARNING] Validation dataset not found: {val_path}")
            print("  Run: python scripts/generate_datasets.py --split val")
            return

        max_samples = self.cfg.validation.get("max_samples")
        self._val_dataloader = build_cached_dataloader(
            path=val_path,
            batch_size=int(self.cfg.validation.batch_size),
            max_samples=int(max_samples) if max_samples else None,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
        )
        print(f"[INFO] Validation dataset loaded: {val_path}")

    def _validation_enabled(self) -> bool:
        """Check if validation is configured and enabled."""
        if not hasattr(self.cfg, "validation"):
            return False
        if not self.cfg.validation.get("enabled", False):
            return False
        return True

    @staticmethod
    def _format_val_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        """Namespace validation metrics for cleaner WandB plots."""
        return {f"val/{key}": value for key, value in metrics.items()}

    @torch.no_grad()
    def validate(self) -> dict[str, Any]:
        """Run validation on the cached validation dataset.

        Returns:
            Dictionary with aggregated validation metrics.
        """
        if self._val_dataloader is None:
            return {}

        self.model.eval()
        bits_per_symbol = int(self.cfg.data.bits_per_symbol)

        total_loss = 0.0
        total_bit_errors = 0
        total_bits = 0
        total_block_errors = 0
        total_blocks = 0
        total_symbol_errors = 0
        total_symbols = 0
        total_channel_mse = 0.0
        num_batches = 0

        for batch in self._val_dataloader:
            batch: CachedNRXBatch
            features = batch.inputs.to(self.device)
            # Flatten bit_labels to match training format
            targets = batch.bit_labels.flatten(start_dim=1).to(self.device)
            channel_targets = batch.channel_target.to(self.device)

            outputs = self.model(features)
            logits = outputs["logits"]
            channel_estimate = outputs["channel_estimate"]

            # Compute loss
            loss = self._compute_loss(outputs, targets, channel_targets)
            total_loss += float(loss.item())

            # Compute metrics
            predictions = (logits > 0.0).to(dtype=targets.dtype)
            bit_errors = (predictions != targets).to(dtype=torch.float32)

            total_bit_errors += int(bit_errors.sum().item())
            total_bits += int(targets.numel())

            block_errors = bit_errors.any(dim=1).to(dtype=torch.float32)
            total_block_errors += int(block_errors.sum().item())
            total_blocks += int(block_errors.numel())

            symbol_errors = (
                bit_errors.view(bit_errors.shape[0], -1, bits_per_symbol).any(dim=-1).to(dtype=torch.float32)
            )
            total_symbol_errors += int(symbol_errors.sum().item())
            total_symbols += int(symbol_errors.numel())

            channel_mse = self.channel_loss_fn(channel_estimate, channel_targets)
            total_channel_mse += float(channel_mse.item())

            num_batches += 1

        self.model.train()

        return {
            "loss": total_loss / max(num_batches, 1),
            "ber": total_bit_errors / max(total_bits, 1),
            "bler": total_block_errors / max(total_blocks, 1),
            "ser": total_symbol_errors / max(total_symbols, 1),
            "channel_mse": total_channel_mse / max(num_batches, 1),
            "num_samples": total_blocks,
        }
