"""Training loop for neural receiver models."""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.data import CachedNRXBatch, build_cached_dataloader
from src.models import build_model_from_config
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
        self.model = self._build_model()

        resume_ckpt = self._load_resume_checkpoint()
        if resume_ckpt is not None:
            self.model.load_state_dict(resume_ckpt["model_state_dict"])
            self.global_step = int(resume_ckpt["global_step"])
            print(f"[RESUME] loaded checkpoint from step {self.global_step}")
        else:
            self._apply_warm_start()
            self.global_step = 0

        unfreeze_at = int(cfg.training.get("unfreeze_experts_at_step", 0))
        if self.global_step < unfreeze_at:
            self._apply_freeze_config()

        self.model = self.model.to(self.device)
        self.num_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self._amp_dtype = self._resolve_amp_dtype(cfg.training.get("mixed_precision"))
        if (
            self.device.type == "cuda"
            and self._amp_dtype == torch.bfloat16
            and hasattr(torch.cuda, "is_bf16_supported")
            and not torch.cuda.is_bf16_supported()
        ):
            print("[WARNING] bf16 AMP requested but not supported on this GPU, disabling mixed precision")
            self._amp_dtype = None
        self._amp_enabled = self.device.type == "cuda" and self._amp_dtype is not None
        scaler_enabled = self._amp_enabled and self._amp_dtype == torch.float16
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.grad_scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        else:  # pragma: no cover - compatibility path
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(cfg.training.learning_rate),
            weight_decay=float(cfg.training.weight_decay),
        )

        if resume_ckpt is not None:
            try:
                self.optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
                for state in self.optimizer.state.values():
                    for v in state.values():
                        if isinstance(v, torch.Tensor):
                            v.data = v.data.to(self.device)
                print("[RESUME] restored optimizer state")
            except (ValueError, KeyError):
                print("[RESUME] optimizer state mismatch, using fresh optimizer")
            scaler_sd = resume_ckpt.get("grad_scaler_state_dict")
            if scaler_sd and self.grad_scaler.is_enabled():
                self.grad_scaler.load_state_dict(scaler_sd)

        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.channel_loss_fn = torch.nn.MSELoss()
        self._val_dataloaders: dict[str, DataLoader] = {}
        self._best_checkpoint_score: float | None = None
        self._train_metrics_ema_alpha = float(cfg.logging.get("train_metrics_ema_alpha", 0.1))
        self._train_metrics_window_size = max(int(cfg.logging.get("train_metrics_window_size", 50)), 1)
        self._train_metric_ema: dict[str, float] = {}
        self._train_metric_windows: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self._train_metrics_window_size)
        )
        self._profile_metric_ema: dict[str, dict[str, float]] = defaultdict(dict)
        self._profile_metric_windows: dict[str, dict[str, deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self._train_metrics_window_size))
        )
        self._setup_validation()
        self._log_model_metadata()

    def train_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        *,
        compute_error_metrics: bool = True,
    ) -> dict[str, Any]:
        """Run a single optimization step."""
        self.model.train()
        features, targets, channel_targets, channel_profile = batch
        features = features.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        channel_targets = channel_targets.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast_context():
            outputs = self.model(features)
            loss = self._compute_loss(outputs, targets, channel_targets)

        if self.grad_scaler.is_enabled():
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
        else:
            loss.backward()

        clip_value = float(self.cfg.training.gradient_clip_norm)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value))
        if self.grad_scaler.is_enabled():
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        metrics: dict[str, Any] = {
            "loss": float(loss.detach().cpu().item()),
            "grad_norm": grad_norm,
            "channel_profile": str(channel_profile),
            **self._collect_aux_metrics(outputs),
        }

        if compute_error_metrics:
            logits = outputs["logits"].float()
            channel_estimate = outputs["channel_estimate"].float()
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
            metrics.update(
                {
                    "ber": float(error_metrics.ber.detach().cpu().item()),
                    "bler": float(error_metrics.bler.detach().cpu().item()),
                    "ser": float(error_metrics.ser.detach().cpu().item()),
                    "block_bit_errors_mean": float(bit_errors_per_block.mean().detach().cpu().item()),
                    "block_bit_errors_median": float(block_error_quantiles[0].detach().cpu().item()),
                    "block_bit_errors_p90": float(block_error_quantiles[1].detach().cpu().item()),
                    "block_bit_errors_max": float(bit_errors_per_block.max().detach().cpu().item()),
                    "channel_mse": float(channel_mse.detach().cpu().item()),
                    "block_bit_errors_hist": wandb.Histogram(bit_errors_per_block.detach().cpu().numpy()),
                }
            )

        return metrics

    def train(self, train_loader, num_steps: int) -> None:
        """Main training loop with periodic metric logging and validation."""
        data_iterator = iter(train_loader)
        progress = tqdm(total=num_steps, initial=self.global_step, desc="train", dynamic_ncols=True)
        self.scheduler = self._build_scheduler(num_steps)
        if self.scheduler is not None and self.global_step > 0:
            for _ in range(self.global_step):
                self.scheduler.step()

        val_enabled = bool(self._val_dataloaders)
        val_every_n = int(self.cfg.validation.every_n_steps) if val_enabled else 0
        checkpoint_cfg = self.cfg.training.checkpoint
        checkpoint_every_n = int(checkpoint_cfg.every_n_steps)
        overfit_enabled = bool(self.cfg.training.get("overfit_single_batch", False))
        overfit_stop_loss = float(self.cfg.training.get("overfit_stop_loss", 0.01))
        overfit_min_steps = int(self.cfg.training.get("overfit_min_steps", 100))
        unfreeze_at_step = int(self.cfg.training.get("unfreeze_experts_at_step", 0))

        while self.global_step < num_steps:
            if unfreeze_at_step > 0 and self.global_step == unfreeze_at_step:
                self._unfreeze_experts()

            fetch_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_loader)
                batch = next(data_iterator)
            data_time = time.perf_counter() - fetch_start

            log_every_n = int(self.cfg.logging.log_every_n_steps)
            next_step = self.global_step + 1
            should_log = next_step == 1 or next_step % log_every_n == 0

            step_start = time.perf_counter()
            metrics = self.train_step(batch, compute_error_metrics=should_log)
            step_time = time.perf_counter() - step_start
            iter_time = data_time + step_time
            batch_size = int(batch[0].shape[0])
            samples_per_s = batch_size / iter_time if iter_time > 0 else 0.0
            self.global_step += 1
            metrics["data_time_s"] = data_time
            metrics["step_time_s"] = step_time
            metrics["iter_time_s"] = iter_time
            metrics["samples_per_s"] = samples_per_s
            train_metrics = self._build_smoothed_train_metrics(metrics)

            progress.update(1)
            progress.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                ber=f"{self._train_metric_ema.get('ber', 0.0):.4f}",
                bler=f"{self._train_metric_ema.get('bler', 0.0):.4f}",
            )

            metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
            train_metrics["lr"] = metrics["lr"]
            if self.use_wandb:
                log_metrics(self._format_logging_metrics(metrics, train_metrics), step=self.global_step)
            else:
                metrics.pop("block_bit_errors_hist", None)

            if self.global_step == 1 or self.global_step % int(self.cfg.logging.log_every_n_steps) == 0:
                profile = str(metrics["channel_profile"])
                compute_summary = ""
                if "expected_flops_ratio" in metrics:
                    compute_summary = (
                        f" exp_flops={metrics['expected_flops_ratio']:.3f}"
                        f" real_flops={metrics['realized_flops_ratio']:.3f}"
                    )
                print(
                    f"step={self.global_step} loss={metrics['loss']:.4f} "
                    f"ema_loss={train_metrics['ema/loss']:.4f} "
                    f"data_t={metrics['data_time_s']:.3f}s step_t={metrics['step_time_s']:.3f}s "
                    f"iter_t={metrics['iter_time_s']:.3f}s samples_s={metrics['samples_per_s']:.1f} "
                    f"ber={metrics['ber']:.4f} ser={metrics['ser']:.4f} bler={metrics['bler']:.4f} "
                    f"profile={profile} "
                    f"{compute_summary}"
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
                self.save_checkpoint(str(latest_path), log_artifact=True, checkpoint_kind="latest")
                print(f"  [CKPT] step={self.global_step} latest -> {latest_path}")

            if overfit_enabled and self.global_step >= overfit_min_steps and metrics["loss"] <= overfit_stop_loss:
                print(
                    f"  [OVERFIT] step={self.global_step} reached target loss "
                    f"{metrics['loss']:.6f} <= {overfit_stop_loss:.6f}"
                )
                break

        progress.close()

    @staticmethod
    def _mean(values: deque[float]) -> float:
        return sum(values) / max(len(values), 1)

    def _build_smoothed_train_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        smoothed: dict[str, Any] = {}
        numeric_metric_keys = [
            key for key, value in metrics.items() if isinstance(value, int | float) and key not in {"channel_profile"}
        ]

        for key in numeric_metric_keys:
            value = float(metrics[key])
            previous = self._train_metric_ema.get(key, value)
            ema_value = previous + self._train_metrics_ema_alpha * (value - previous)
            self._train_metric_ema[key] = ema_value
            smoothed[f"ema/{key}"] = ema_value

        profile = str(metrics["channel_profile"])
        for key in numeric_metric_keys:
            value = float(metrics[key])
            previous = self._profile_metric_ema[profile].get(key, value)
            ema_value = previous + self._train_metrics_ema_alpha * (value - previous)
            self._profile_metric_ema[profile][key] = ema_value
            smoothed[f"profile/{profile}/ema/{key}"] = ema_value

        return smoothed

    @staticmethod
    def _format_logging_metrics(metrics: dict[str, Any], smoothed_metrics: dict[str, Any]) -> dict[str, Any]:
        """Namespace training metrics for cleaner WandB plots."""

        logged_metrics = {f"train/{key}": value for key, value in metrics.items() if key != "channel_profile"}
        logged_metrics.update({f"train/{key}": value for key, value in smoothed_metrics.items()})
        logged_metrics["train/channel_profile"] = str(metrics["channel_profile"])
        return logged_metrics

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
                "grad_scaler_state_dict": self.grad_scaler.state_dict() if self.grad_scaler.is_enabled() else None,
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
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

    def _build_scheduler(self, num_steps: int) -> torch.optim.lr_scheduler.LRScheduler | None:
        scheduler_cfg = self.cfg.training.get("scheduler")
        if scheduler_cfg is None:
            return None

        scheduler_name = str(scheduler_cfg.get("name", "none")).lower()
        if scheduler_name in {"", "none", "off", "disabled", "null"}:
            return None

        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(int(num_steps), 1),
                eta_min=float(scheduler_cfg.get("min_lr", 0.0)),
            )

        raise ValueError(f"Unsupported training.scheduler.name: {scheduler_name}")

    def _load_resume_checkpoint(self) -> dict[str, Any] | None:
        """Load a full training checkpoint for resumption if configured."""
        resume_source = str(self.cfg.training.get("resume_from", "") or "")
        if not resume_source:
            return None

        path = Path(resume_source)
        if not path.exists():
            import wandb as _wandb

            api = _wandb.Api()
            artifact = api.artifact(resume_source)
            artifact_dir = Path(artifact.download())
            pt_files = sorted(artifact_dir.glob("*.pt"))
            if not pt_files:
                raise FileNotFoundError(f"No .pt file in resume artifact {resume_source!r}")
            path = pt_files[0]

        ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt

    def _apply_warm_start(self) -> None:
        """Load dense checkpoint weights into MoE stem and expert branches if configured."""
        warm_cfg = self.cfg.training.get("warm_start")
        if not warm_cfg:
            return
        from src.models.warm_start import load_warm_start

        stem_ckpt = str(warm_cfg.get("stem_checkpoint", ""))
        experts_cfg = warm_cfg.get("experts") or {}
        if not stem_ckpt or not experts_cfg:
            raise ValueError("training.warm_start requires stem_checkpoint and experts mapping")
        load_warm_start(
            self.model,
            stem_checkpoint=stem_ckpt,
            expert_checkpoints={str(k): str(v) for k, v in experts_cfg.items()},
            device="cpu",
        )

    def _apply_freeze_config(self) -> None:
        """Freeze all expert parameters if training.freeze_experts=true (Stage 2B)."""
        if not bool(self.cfg.training.get("freeze_experts", False)):
            return
        if not hasattr(self.model, "experts"):
            print("[WARNING] freeze_experts=true but model has no .experts attribute — ignoring")
            return
        for param in self.model.experts.parameters():
            param.requires_grad_(False)
        frozen = sum(p.numel() for p in self.model.experts.parameters())
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[INFO] Frozen {frozen:,} expert params ({100 * frozen / max(total, 1):.1f}% of {total:,} total)")

    def _unfreeze_experts(self) -> None:
        """Unfreeze expert parameters and add them to the optimizer (Stage 2C transition)."""
        if not hasattr(self.model, "experts"):
            return
        newly_trainable = [p for p in self.model.experts.parameters() if not p.requires_grad]
        for p in newly_trainable:
            p.requires_grad_(True)
        if newly_trainable:
            self.optimizer.add_param_group(
                {
                    "params": newly_trainable,
                    "lr": float(self.cfg.training.learning_rate),
                    "weight_decay": float(self.cfg.training.weight_decay),
                }
            )
            print(
                f"  [UNFREEZE] step={self.global_step}: "
                f"unfroze {sum(p.numel() for p in newly_trainable):,} expert params — "
                "joint fine-tune begins"
            )

    def cleanup(self):
        """Cleanup and finish logging."""
        finish_wandb()

    def _build_model(self) -> torch.nn.Module:
        return build_model_from_config(self.cfg)

    def _log_model_metadata(self) -> None:
        if wandb.run is None:
            return

        wandb.run.summary["model/num_parameters"] = self.num_parameters
        wandb.run.summary["model/family"] = str(self.cfg.model.family)
        wandb.run.summary["model/name"] = str(self.cfg.model.name)
        wandb.run.summary["runtime/device"] = str(self.device)
        wandb.run.summary["runtime/mixed_precision"] = self._mixed_precision_mode()
        if hasattr(self.model, "expert_names"):
            wandb.run.summary["model/expert_names"] = list(getattr(self.model, "expert_names"))

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

    @staticmethod
    def _resolve_amp_dtype(mode: object) -> torch.dtype | None:
        if mode is None:
            return None
        normalized = str(mode).strip().lower()
        if normalized in {"", "none", "null", "off", "false"}:
            return None
        if normalized == "fp16":
            return torch.float16
        if normalized == "bf16":
            return torch.bfloat16
        raise ValueError(f"Unsupported training.mixed_precision: {mode}")

    def _mixed_precision_mode(self) -> str:
        if not self._amp_enabled or self._amp_dtype is None:
            return "off"
        if self._amp_dtype == torch.float16:
            return "fp16"
        if self._amp_dtype == torch.bfloat16:
            return "bf16"
        return str(self._amp_dtype)

    def _autocast_context(self):
        if not self._amp_enabled or self._amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self._amp_dtype)

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
        compute_cfg = self.cfg.model.compute
        flops_penalty_alpha = float(compute_cfg.get("flops_penalty_alpha", 0.0))
        if flops_penalty_alpha > 0.0 and "expected_flops_ratio" in outputs:
            final_loss = final_loss + flops_penalty_alpha * outputs["expected_flops_ratio"]

        load_balance_beta = float(compute_cfg.get("load_balance_beta", 0.0))
        if load_balance_beta > 0.0 and "router_probs" in outputs:
            mean_probs = outputs["router_probs"].mean(dim=0)
            uniform_probs = torch.full_like(mean_probs, 1.0 / max(mean_probs.numel(), 1))
            load_balance_penalty = torch.mean((mean_probs - uniform_probs) ** 2)
            final_loss = final_loss + load_balance_beta * load_balance_penalty

        # Switch Transformer auxiliary loss: n_experts * sum(f_i * P_i)
        # f_i = fraction of tokens hard-routed to expert i (detached)
        # P_i = mean soft router probability for expert i (differentiable)
        switch_aux_weight = float(compute_cfg.get("switch_aux_weight", 0.0))
        if switch_aux_weight > 0.0 and "router_probs" in outputs and "selected_expert_index" in outputs:
            rp = outputs["router_probs"]  # [batch, n_experts]
            sel = outputs["selected_expert_index"]  # [batch]
            n_exp = rp.shape[-1]
            f = torch.stack([(sel == i).float().mean() for i in range(n_exp)]).detach()
            P = rp.mean(dim=0)
            final_loss = final_loss + switch_aux_weight * n_exp * (f * P).sum()

        # Soft capacity penalty: penalise any expert exceeding capacity_factor / n_experts
        # capacity_factor=1.5 → max 50% per expert for 3 experts
        capacity_weight = float(compute_cfg.get("capacity_weight", 0.0))
        capacity_factor = float(compute_cfg.get("capacity_factor", 1.5))
        if capacity_weight > 0.0 and "router_probs" in outputs:
            mean_probs = outputs["router_probs"].mean(dim=0)
            max_allowed = capacity_factor / max(mean_probs.numel(), 1)
            final_loss = final_loss + capacity_weight * torch.clamp(mean_probs - max_allowed, min=0.0).sum()

        intermediate_logits = outputs["intermediate_logits"]
        intermediate_channel_estimates = outputs["intermediate_channel_estimates"]
        if not intermediate_logits or not intermediate_channel_estimates:
            return final_loss

        multi_loss = torch.zeros((), device=self.device)
        for logits, channel_estimate in zip(intermediate_logits, intermediate_channel_estimates):
            multi_loss = multi_loss + self.loss_fn(logits, targets)
            multi_loss = multi_loss + channel_weight * self.channel_loss_fn(channel_estimate, channel_targets)
        return final_loss + multi_loss

    def _collect_aux_metrics(self, outputs: dict[str, torch.Tensor | list[torch.Tensor]]) -> dict[str, float]:
        aux_metrics: dict[str, float] = {}
        router_probs = outputs.get("router_probs")
        if isinstance(router_probs, torch.Tensor):
            entropy = -(router_probs * router_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
            aux_metrics["router_entropy"] = float(entropy.detach().cpu().item())
            mean_probs = router_probs.mean(dim=0)
            expert_names = getattr(self.model, "expert_names", [str(index) for index in range(mean_probs.numel())])
            for expert_index, expert_name in enumerate(expert_names):
                aux_metrics[f"expert_usage/{expert_name}"] = float(mean_probs[expert_index].detach().cpu().item())

        for metric_name in [
            "expected_flops",
            "expected_flops_ratio",
            "realized_flops",
            "realized_flops_ratio",
        ]:
            metric_value = outputs.get(metric_name)
            if isinstance(metric_value, torch.Tensor):
                aux_metrics[metric_name] = float(metric_value.detach().cpu().item())
        return aux_metrics

    def _setup_validation(self) -> None:
        """Initialize validation dataloader if enabled."""
        if not self._validation_enabled():
            return

        validation_cfg = self.cfg.validation
        max_samples = validation_cfg.get("max_samples")
        batch_size = int(validation_cfg.batch_size)
        loader_kwargs = dict(
            batch_size=batch_size,
            max_samples=int(max_samples) if max_samples else None,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
        )

        hf_dataset = validation_cfg.get("hf_dataset")
        profiles = validation_cfg.get("profiles")

        if hf_dataset:
            hf_split = str(validation_cfg.get("hf_split", "val"))
            for profile in profiles or []:
                self._val_dataloaders[str(profile)] = build_cached_dataloader(
                    hf_repo=str(hf_dataset),
                    hf_config=str(profile),
                    hf_split=hf_split,
                    **loader_kwargs,
                )
                print(f"[INFO] Validation dataset loaded: {hf_dataset}/{profile} (split={hf_split})")
        else:
            dataset_specs: list[tuple[str, Path]] = []
            if profiles:
                data_dir = Path(str(validation_cfg.get("data_dir", "data/val")))
                dataset_specs.extend((str(profile), data_dir / f"{profile}.pt") for profile in profiles)
            elif validation_cfg.get("dataset_path"):
                val_path = Path(str(validation_cfg.dataset_path))
                dataset_specs.append((val_path.stem, val_path))

            for name, path in dataset_specs:
                if not path.exists():
                    print(f"[WARNING] Validation dataset not found: {path}")
                    continue
                self._val_dataloaders[name] = build_cached_dataloader(path=path, **loader_kwargs)
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

        # Per-sample accumulators for SNR-binned metrics
        all_snr: list[torch.Tensor] = []
        all_ber: list[torch.Tensor] = []
        all_bler: list[torch.Tensor] = []
        all_ser: list[torch.Tensor] = []

        for batch in dataloader:
            batch: CachedNRXBatch
            features = batch.inputs.to(self.device)
            targets = batch.bit_labels.flatten(start_dim=1).to(self.device)
            channel_targets = batch.channel_target.to(self.device)

            with self._autocast_context():
                outputs = self.model(features)
                loss = self._compute_loss(outputs, targets, channel_targets)

            logits = outputs["logits"].float()
            channel_estimate = outputs["channel_estimate"].float()
            total_loss += float(loss.item())

            error_metrics = compute_batch_error_metrics(
                logits,
                targets,
                bits_per_symbol=int(self.cfg.data.bits_per_symbol),
                num_subcarriers=int(self.cfg.data.num_subcarriers),
                num_ofdm_symbols=int(self.cfg.data.num_ofdm_symbols),
            )

            total_bit_errors += int(error_metrics.bit_errors.sum().item())
            total_bits += int(targets.numel())
            total_block_errors += int(error_metrics.block_errors.sum().item())
            total_blocks += int(error_metrics.block_errors.numel())
            total_symbol_errors += int(error_metrics.symbol_errors.sum().item())
            total_symbols += int(error_metrics.symbol_errors.numel())

            channel_mse = self.channel_loss_fn(channel_estimate, channel_targets)
            total_channel_mse += float(channel_mse.item())
            num_batches += 1

            # Accumulate per-sample metrics for SNR binning
            all_snr.append(batch.snr_db.cpu())
            all_ber.append(error_metrics.bit_errors.float().mean(dim=1).cpu())
            all_bler.append(error_metrics.block_errors.float().cpu())
            all_ser.append(error_metrics.symbol_errors.float().mean(dim=(1, 2)).cpu())

        metrics: dict[str, Any] = {
            "loss": total_loss / max(num_batches, 1),
            "ber": total_bit_errors / max(total_bits, 1),
            "bler": total_block_errors / max(total_blocks, 1),
            "ser": total_symbol_errors / max(total_symbols, 1),
            "channel_mse": total_channel_mse / max(num_batches, 1),
            "num_samples": total_blocks,
        }

        num_snr_bins = int(self.cfg.validation.get("snr_bins", 0))
        if num_snr_bins > 0 and all_snr:
            snr = torch.cat(all_snr)
            ber = torch.cat(all_ber)
            bler = torch.cat(all_bler)
            ser = torch.cat(all_ser)
            edges = torch.linspace(float(snr.min()), float(snr.max()), num_snr_bins + 1)
            for i in range(num_snr_bins):
                lo, hi = float(edges[i]), float(edges[i + 1])
                mask = (snr >= lo) & (snr <= hi if i == num_snr_bins - 1 else snr < hi)
                if not mask.any():
                    continue
                center = round((lo + hi) / 2.0, 1)
                metrics[f"snr_bin_{i}/ber"] = float(ber[mask].mean())
                metrics[f"snr_bin_{i}/bler"] = float(bler[mask].mean())
                metrics[f"snr_bin_{i}/ser"] = float(ser[mask].mean())
                metrics[f"snr_bin_{i}/snr_center"] = center

        return metrics

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
