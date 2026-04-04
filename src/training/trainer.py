"""Training loop for dense neural receiver baselines."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.data import CachedNRXBatch, build_cached_dataloader
from src.models import StaticDenseNRX
from src.utils.logging import finish_wandb, log_metrics, setup_wandb
from src.utils.metrics import compute_batch_error_metrics
from src.utils.wandb_artifacts import build_checkpoint_artifact_ref, log_checkpoint_artifact


class Trainer:
    """Simple trainer for neural receiver models."""

    def __init__(self, cfg: Any, job_id: str | None = None):
        self.cfg = cfg
        self.job_id = job_id
        self.device = self._resolve_device(str(cfg.runtime.device))
        self.use_wandb = setup_wandb(cfg, job_id=job_id, job_type="train")
        self.model = self._build_model().to(self.device)
        self.num_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.training.learning_rate),
            weight_decay=float(cfg.training.weight_decay),
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.channel_loss_fn = torch.nn.MSELoss()
        self.global_step = 0
        self._val_dataloaders: dict[str, DataLoader] = {}
        self._best_checkpoint_score: float | None = None
        self._setup_validation()
        self._log_model_metadata()

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

        error_metrics = compute_batch_error_metrics(
            logits,
            targets,
            bits_per_symbol=int(self.cfg.data.bits_per_symbol),
            num_subcarriers=int(self.cfg.data.num_subcarriers),
            num_ofdm_symbols=int(self.cfg.data.num_ofdm_symbols),
        )
        bit_errors_per_block = error_metrics.bit_errors_per_block

        block_error_quantiles = torch.quantile(bit_errors_per_block, torch.tensor([0.5, 0.9], device=self.device))
        channel_mse = self.channel_loss_fn(channel_estimate, channel_targets)

        return {
            "loss": float(loss.detach().cpu().item()),
            "ber": float(error_metrics.ber.detach().cpu().item()),
            "bler": float(error_metrics.bler.detach().cpu().item()),
            "ser": float(error_metrics.ser.detach().cpu().item()),
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

        val_enabled = bool(self._val_dataloaders)
        val_every_n = int(self.cfg.validation.every_n_steps) if val_enabled else 0
        checkpoint_cfg = self.cfg.training.checkpoint
        checkpoint_every_n = int(checkpoint_cfg.every_n_steps)
        overfit_enabled = bool(self.cfg.training.get("overfit_single_batch", False))
        overfit_stop_loss = float(self.cfg.training.get("overfit_stop_loss", 0.01))
        overfit_min_steps = int(self.cfg.training.get("overfit_min_steps", 100))

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
                summary = " ".join(
                    f"{name}: loss={metrics['loss']:.4f} ber={metrics['ber']:.4f} bler={metrics['bler']:.4f}"
                    for name, metrics in val_metrics.items()
                )
                print(f"  [VAL] step={self.global_step} {summary}")

                if bool(checkpoint_cfg.save_best):
                    self._maybe_save_best_checkpoint(val_metrics)

            if (
                bool(checkpoint_cfg.save_latest)
                and checkpoint_every_n > 0
                and self.global_step % checkpoint_every_n == 0
            ):
                latest_path = self._checkpoint_dir() / f"{self.cfg.model.name}_latest.pt"
                self.save_checkpoint(str(latest_path), log_artifact=False, checkpoint_kind="latest")
                print(f"  [CKPT] step={self.global_step} latest -> {latest_path}")

            if overfit_enabled and self.global_step >= overfit_min_steps and metrics["loss"] <= overfit_stop_loss:
                print(
                    f"  [OVERFIT] step={self.global_step} reached target loss "
                    f"{metrics['loss']:.6f} <= {overfit_stop_loss:.6f}"
                )
                break

        progress.close()

    @staticmethod
    def _format_logging_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        """Namespace training metrics for cleaner WandB plots."""

        return {f"train/{key}": value for key, value in metrics.items()}

    def save_checkpoint(
        self,
        path: str,
        *,
        artifact_aliases: list[str] | None = None,
        artifact_primary_alias: str | None = None,
        log_artifact: bool = True,
        checkpoint_kind: str = "final",
    ) -> str | None:
        """Save model checkpoint and resolved config."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        wandb_info: dict[str, Any] | None = None
        if wandb.run is not None:
            wandb_info = {
                "run_path": wandb.run.path,
            }
            if log_artifact and artifact_primary_alias:
                wandb_info["checkpoint_artifact"] = build_checkpoint_artifact_ref(
                    wandb.run.project,
                    wandb.run.entity,
                    wandb.run.id,
                    alias=artifact_primary_alias,
                    exp_name=self.cfg.experiment.get("exp_name"),
                    model_name=self.cfg.model.name,
                )

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "config": OmegaConf.to_container(self.cfg, resolve=True),
                "checkpoint_kind": checkpoint_kind,
                "wandb": wandb_info,
            },
            checkpoint_path,
        )

        artifact_ref: str | None = None
        if log_artifact:
            alias_list = artifact_aliases or ["latest", checkpoint_kind, f"step-{self.global_step}"]
            artifact_ref = log_checkpoint_artifact(
                checkpoint_path,
                metadata={
                    "global_step": self.global_step,
                    "model_family": str(self.cfg.model.family),
                    "model_name": str(self.cfg.model.name),
                    "num_parameters": self.num_parameters,
                    "experiment_name": self.cfg.experiment.get("exp_name"),
                    "batch_name": self.cfg.experiment.get("batch_name"),
                    "study_slug": self.cfg.experiment.get("study_slug"),
                    "study_path": self.cfg.experiment.get("study_path"),
                    "question": self.cfg.experiment.get("question"),
                    "checkpoint_kind": checkpoint_kind,
                },
                aliases=alias_list,
            )

            if artifact_ref and wandb.run is not None:
                wandb.run.summary[f"artifacts/checkpoint_{checkpoint_kind}"] = artifact_ref
                if checkpoint_kind == "final":
                    wandb.run.summary["artifacts/checkpoint"] = artifact_ref

        if artifact_ref and wandb.run is not None:
            wandb.run.summary["model/num_parameters"] = self.num_parameters
            wandb.run.summary["model/global_step"] = self.global_step
        return artifact_ref

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

    def _log_model_metadata(self) -> None:
        if wandb.run is None:
            return

        wandb.run.summary["model/num_parameters"] = self.num_parameters
        wandb.run.summary["model/family"] = str(self.cfg.model.family)
        wandb.run.summary["model/name"] = str(self.cfg.model.name)
        wandb.run.summary["runtime/device"] = str(self.device)

    def _checkpoint_dir(self) -> Path:
        return Path(str(self.cfg.training.checkpoint_dir))

    def _compute_checkpoint_score(self, val_metrics: dict[str, dict[str, Any]]) -> float | None:
        checkpoint_cfg = self.cfg.training.checkpoint
        metric_name = str(checkpoint_cfg.best_metric)
        aggregation = str(checkpoint_cfg.best_metric_aggregation)
        configured_profiles = checkpoint_cfg.get("best_profiles")
        profile_names = (
            [str(profile) for profile in configured_profiles] if configured_profiles else list(val_metrics.keys())
        )

        values: list[float] = []
        for profile in profile_names:
            metrics = val_metrics.get(profile)
            if metrics is None or metric_name not in metrics:
                continue
            values.append(float(metrics[metric_name]))

        if not values:
            return None

        if aggregation == "mean":
            return sum(values) / len(values)
        if aggregation == "max":
            return max(values)
        if aggregation == "min":
            return min(values)
        raise ValueError(f"Unsupported checkpoint aggregation: {aggregation}")

    def _is_better_checkpoint(self, score: float) -> bool:
        if self._best_checkpoint_score is None:
            return True

        mode = str(self.cfg.training.checkpoint.best_metric_mode)
        if mode == "min":
            return score < self._best_checkpoint_score - 1e-12
        if mode == "max":
            return score > self._best_checkpoint_score + 1e-12
        raise ValueError(f"Unsupported checkpoint mode: {mode}")

    def _maybe_save_best_checkpoint(self, val_metrics: dict[str, dict[str, Any]]) -> None:
        score = self._compute_checkpoint_score(val_metrics)
        if score is None or not math.isfinite(score):
            return

        if not self._is_better_checkpoint(score):
            return

        self._best_checkpoint_score = score
        best_path = self._checkpoint_dir() / f"{self.cfg.model.name}_best.pt"
        self.save_checkpoint(
            str(best_path),
            artifact_aliases=["best", f"step-{self.global_step}"],
            artifact_primary_alias="best",
            log_artifact=True,
            checkpoint_kind="best",
        )

        if wandb.run is not None:
            wandb.run.summary["checkpoint/best_score"] = score
            wandb.run.summary["checkpoint/best_step"] = self.global_step

        print(f"  [CKPT] step={self.global_step} new best -> {best_path} (score={score:.6f})")

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
        return final_loss + multi_loss

    def _setup_validation(self) -> None:
        """Initialize validation dataloader if enabled."""
        if not self._validation_enabled():
            return

        validation_cfg = self.cfg.validation
        dataset_specs: list[tuple[str, Path]] = []

        profiles = validation_cfg.get("profiles")
        if profiles:
            data_dir = Path(str(validation_cfg.get("data_dir", "data/val")))
            dataset_specs.extend((str(profile), data_dir / f"{profile}.pt") for profile in profiles)
        elif validation_cfg.get("dataset_path"):
            val_path = Path(str(validation_cfg.dataset_path))
            dataset_specs.append((val_path.stem, val_path))

        max_samples = validation_cfg.get("max_samples")
        for name, path in dataset_specs:
            if not path.exists():
                print(f"[WARNING] Validation dataset not found: {path}")
                continue

            self._val_dataloaders[name] = build_cached_dataloader(
                path=path,
                batch_size=int(validation_cfg.batch_size),
                max_samples=int(max_samples) if max_samples else None,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
            )
            print(f"[INFO] Validation dataset loaded: {path}")

        if not self._val_dataloaders:
            print("  Run: python scripts/generate_datasets.py generation.split=val")

    def _validation_enabled(self) -> bool:
        """Check if validation is configured and enabled."""
        if not hasattr(self.cfg, "validation"):
            return False
        if not self.cfg.validation.get("enabled", False):
            return False
        return True

    @staticmethod
    def _format_val_metrics(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Namespace validation metrics for cleaner WandB plots."""
        flattened: dict[str, Any] = {}
        for dataset_name, dataset_metrics in metrics.items():
            for key, value in dataset_metrics.items():
                flattened[f"val/{dataset_name}/{key}"] = value
        return flattened

    @torch.no_grad()
    def _validate_dataloader(self, dataloader: DataLoader) -> dict[str, Any]:
        total_loss = 0.0
        total_bit_errors = 0
        total_bits = 0
        total_block_errors = 0
        total_blocks = 0
        total_symbol_errors = 0
        total_symbols = 0
        total_channel_mse = 0.0
        num_batches = 0

        for batch in dataloader:
            batch: CachedNRXBatch
            features = batch.inputs.to(self.device)
            targets = batch.bit_labels.flatten(start_dim=1).to(self.device)
            channel_targets = batch.channel_target.to(self.device)

            outputs = self.model(features)
            logits = outputs["logits"]
            channel_estimate = outputs["channel_estimate"]

            loss = self._compute_loss(outputs, targets, channel_targets)
            total_loss += float(loss.item())

            error_metrics = compute_batch_error_metrics(
                logits,
                targets,
                bits_per_symbol=int(self.cfg.data.bits_per_symbol),
                num_subcarriers=int(self.cfg.data.num_subcarriers),
                num_ofdm_symbols=int(self.cfg.data.num_ofdm_symbols),
            )
            bit_errors = error_metrics.bit_errors

            total_bit_errors += int(bit_errors.sum().item())
            total_bits += int(targets.numel())

            block_errors = error_metrics.block_errors
            total_block_errors += int(block_errors.sum().item())
            total_blocks += int(block_errors.numel())

            symbol_errors = error_metrics.symbol_errors
            total_symbol_errors += int(symbol_errors.sum().item())
            total_symbols += int(symbol_errors.numel())

            channel_mse = self.channel_loss_fn(channel_estimate, channel_targets)
            total_channel_mse += float(channel_mse.item())

            num_batches += 1

        return {
            "loss": total_loss / max(num_batches, 1),
            "ber": total_bit_errors / max(total_bits, 1),
            "bler": total_block_errors / max(total_blocks, 1),
            "ser": total_symbol_errors / max(total_symbols, 1),
            "channel_mse": total_channel_mse / max(num_batches, 1),
            "num_samples": total_blocks,
        }

    @torch.no_grad()
    def validate(self) -> dict[str, dict[str, Any]]:
        """Run validation on the cached validation dataset.

        Returns:
            Dictionary keyed by validation dataset name.
        """
        if not self._val_dataloaders:
            return {}

        self.model.eval()

        metrics = {name: self._validate_dataloader(dataloader) for name, dataloader in self._val_dataloaders.items()}

        self.model.train()
        return metrics
