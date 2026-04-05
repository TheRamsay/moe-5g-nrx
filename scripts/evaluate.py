#!/usr/bin/env python


from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CachedNRXBatch, build_cached_dataloader  # noqa: E402
from src.models import build_model_from_config  # noqa: E402
from src.utils.metrics import compute_batch_error_metrics  # noqa: E402
from src.utils.wandb_artifacts import build_dataset_artifact_name, use_artifact_path  # noqa: E402


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = build_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config


def _download_checkpoint_artifact(artifact_ref: str) -> Path:
    artifact = wandb.Api().artifact(artifact_ref)
    download_dir = Path(artifact.download())
    files = [path for path in download_dir.iterdir() if path.is_file()]
    if len(files) == 1:
        return files[0]
    raise FileNotFoundError(f"Unable to resolve checkpoint file in {download_dir}")


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    bits_per_symbol: int,
    channel_loss_weight: float,
    snr_bins: int | None = None,
    failure_examples_limit: int = 20,
) -> dict[str, Any]:
    """Evaluate model on a dataset.

    Args:
        model: Trained model.
        dataloader: DataLoader for test dataset.
        device: Device to run evaluation on.
        bits_per_symbol: Number of bits per symbol (for SER calculation).
        snr_bins: If provided, compute metrics per SNR bin.

    Returns:
        Dictionary with evaluation metrics.
    """
    loss_fn = torch.nn.BCEWithLogitsLoss()
    channel_loss_fn = torch.nn.MSELoss()

    # Accumulators for overall metrics
    total_loss = 0.0
    total_bce_loss = 0.0
    total_bit_errors = 0
    total_bits = 0
    total_block_errors = 0
    total_blocks = 0
    total_symbol_errors = 0
    total_symbols = 0
    total_channel_mse = 0.0
    num_batches = 0

    # SNR-binned accumulators
    all_snr_db: list[float] = []
    all_bit_errors: list[int] = []
    all_bits_per_sample: list[int] = []
    all_block_errors: list[int] = []
    failure_examples: list[dict[str, Any]] = []
    failure_grid_error_sum: np.ndarray | None = None
    failure_grid_failed_blocks = 0
    all_selected_experts: list[int] = []
    all_expected_flops: list[float] = []
    all_realized_flops: list[float] = []
    sample_offset = 0

    expert_names = list(getattr(model, "expert_names", []))
    base_flops = float(getattr(model, "base_flops", torch.tensor(0.0)).item())
    expert_flops_tensor = getattr(model, "expert_flops", None)
    expert_flops: np.ndarray | None = None
    if isinstance(expert_flops_tensor, torch.Tensor):
        expert_flops = expert_flops_tensor.detach().cpu().numpy().astype(np.float64)

    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
        batch: CachedNRXBatch
        features = batch.inputs.to(device)
        targets = batch.bit_labels.flatten(start_dim=1).to(device)
        channel_targets = batch.channel_target.to(device)
        snr_db = batch.snr_db

        outputs = model(features)
        logits = outputs["logits"]
        channel_estimate = outputs["channel_estimate"]

        selected_expert_index = outputs.get("selected_expert_index")
        if isinstance(selected_expert_index, torch.Tensor):
            selected_np = selected_expert_index.detach().cpu().numpy().astype(int)
            all_selected_experts.extend(selected_np.tolist())
            if expert_flops is not None:
                all_realized_flops.extend((base_flops + expert_flops[selected_np]).tolist())

        router_probs = outputs.get("router_probs")
        if isinstance(router_probs, torch.Tensor) and expert_flops is not None:
            expected_flops = base_flops + torch.sum(
                router_probs * torch.as_tensor(expert_flops, device=router_probs.device), dim=-1
            )
            all_expected_flops.extend(expected_flops.detach().cpu().numpy().astype(np.float64).tolist())

        # Loss
        bce_loss = loss_fn(logits, targets)
        channel_mse = channel_loss_fn(channel_estimate, channel_targets)
        loss = bce_loss + channel_loss_weight * channel_mse
        total_loss += float(loss.item())
        total_bce_loss += float(bce_loss.item())

        # Compute per-sample metrics
        error_metrics = compute_batch_error_metrics(
            logits,
            targets,
            bits_per_symbol=bits_per_symbol,
            num_subcarriers=features.shape[2],
            num_ofdm_symbols=features.shape[3],
        )
        bit_errors = error_metrics.bit_errors
        bit_errors_grid = bit_errors.reshape(features.shape[0], bits_per_symbol, features.shape[2], features.shape[3])
        symbol_errors_grid = bit_errors_grid.any(dim=1).to(dtype=torch.float32)

        # Per-sample bit errors (for SNR analysis)
        per_sample_bit_errors = bit_errors.sum(dim=1).cpu().numpy()
        per_sample_bits = targets.shape[1]
        per_sample_block_error = error_metrics.block_errors.cpu().numpy().astype(int)
        sample_ber = per_sample_bit_errors / max(per_sample_bits, 1)

        all_snr_db.extend(snr_db.numpy().tolist())
        all_bit_errors.extend(per_sample_bit_errors.tolist())
        all_bits_per_sample.extend([per_sample_bits] * len(snr_db))
        all_block_errors.extend(per_sample_block_error.tolist())

        failed_mask = error_metrics.block_errors.to(dtype=torch.bool)
        failed_blocks = int(failed_mask.sum().item())
        if failed_blocks > 0:
            failed_symbol_errors = symbol_errors_grid[failed_mask].sum(dim=0).cpu().numpy()
            if failure_grid_error_sum is None:
                failure_grid_error_sum = np.zeros_like(failed_symbol_errors, dtype=np.float64)
            failure_grid_error_sum += failed_symbol_errors
            failure_grid_failed_blocks += failed_blocks

        if failure_examples_limit > 0:
            for idx, (sample_snr, sample_errors, sample_block_error, sample_ber_value) in enumerate(
                zip(
                    snr_db.numpy().tolist(),
                    per_sample_bit_errors.tolist(),
                    per_sample_block_error.tolist(),
                    sample_ber.tolist(),
                )
            ):
                failure_examples.append(
                    {
                        "sample_index": sample_offset + idx,
                        "snr_db": float(sample_snr),
                        "bit_errors": int(sample_errors),
                        "bits_per_sample": int(per_sample_bits),
                        "ber": float(sample_ber_value),
                        "block_error": int(sample_block_error),
                    }
                )

        sample_offset += len(snr_db)

        # Aggregate metrics
        total_bit_errors += int(bit_errors.sum().item())
        total_bits += int(targets.numel())

        block_errors = error_metrics.block_errors
        total_block_errors += int(block_errors.sum().item())
        total_blocks += int(block_errors.numel())

        symbol_errors = error_metrics.symbol_errors
        total_symbol_errors += int(symbol_errors.sum().item())
        total_symbols += int(symbol_errors.numel())

        total_channel_mse += float(channel_mse.item())

        num_batches += 1

    results = {
        "loss": total_loss / max(num_batches, 1),
        "loss_total": total_loss / max(num_batches, 1),
        "loss_bce": total_bce_loss / max(num_batches, 1),
        "ber": total_bit_errors / max(total_bits, 1),
        "bler": total_block_errors / max(total_blocks, 1),
        "ser": total_symbol_errors / max(total_symbols, 1),
        "channel_mse": total_channel_mse / max(num_batches, 1),
        "num_samples": total_blocks,
        "num_bit_errors": total_bit_errors,
        "num_block_errors": total_block_errors,
    }

    # SNR-binned analysis
    if snr_bins is not None and snr_bins > 0:
        snr_array = np.array(all_snr_db)
        bit_errors_array = np.array(all_bit_errors)
        bits_per_sample_array = np.array(all_bits_per_sample)
        block_errors_array = np.array(all_block_errors)

        # Create SNR bins
        snr_min, snr_max = snr_array.min(), snr_array.max()
        bin_edges = np.linspace(snr_min, snr_max, snr_bins + 1)
        bin_indices = np.digitize(snr_array, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, snr_bins - 1)

        snr_binned = {}
        snr_binned_rows: list[dict[str, Any]] = []
        for i in range(snr_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue

            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            bin_ber = bit_errors_array[mask].sum() / bits_per_sample_array[mask].sum()
            bin_bler = block_errors_array[mask].mean()

            snr_binned[f"bin_{i}"] = {
                "snr_center": float(bin_center),
                "snr_range": (float(bin_edges[i]), float(bin_edges[i + 1])),
                "ber": float(bin_ber),
                "bler": float(bin_bler),
                "num_samples": int(mask.sum()),
            }
            snr_binned_rows.append(
                {
                    "bin_index": i,
                    "snr_min": float(bin_edges[i]),
                    "snr_max": float(bin_edges[i + 1]),
                    "snr_center": float(bin_center),
                    "ber": float(bin_ber),
                    "bler": float(bin_bler),
                    "num_samples": int(mask.sum()),
                }
            )

            if all_selected_experts and expert_names:
                selected_array = np.array(all_selected_experts)
                expert_counts = np.bincount(selected_array[mask], minlength=len(expert_names))
                snr_binned_rows[-1]["expert_usage"] = {
                    expert_name: float(expert_counts[index] / max(mask.sum(), 1))
                    for index, expert_name in enumerate(expert_names)
                }

        results["snr_binned"] = snr_binned
        results["snr_binned_rows"] = snr_binned_rows

    if failure_examples_limit > 0:
        failure_examples_sorted = sorted(
            failure_examples,
            key=lambda row: (row["bit_errors"], row["block_error"], -row["snr_db"]),
            reverse=True,
        )
        results["failure_examples"] = failure_examples_sorted[:failure_examples_limit]

    if failure_grid_error_sum is not None and failure_grid_failed_blocks > 0:
        failure_grid = failure_grid_error_sum / failure_grid_failed_blocks
        results["failure_grid"] = failure_grid
        results["failure_grid_rows"] = [
            {
                "subcarrier_index": subcarrier_index,
                "ofdm_symbol_index": ofdm_symbol_index,
                "symbol_error_rate": float(failure_grid[subcarrier_index, ofdm_symbol_index]),
                "failed_block_count": failure_grid_failed_blocks,
            }
            for subcarrier_index in range(failure_grid.shape[0])
            for ofdm_symbol_index in range(failure_grid.shape[1])
        ]

    if all_expected_flops:
        results["expected_flops"] = float(np.mean(all_expected_flops))
    if all_realized_flops:
        results["realized_flops"] = float(np.mean(all_realized_flops))
    if all_selected_experts and expert_names:
        selected_array = np.array(all_selected_experts)
        expert_counts = np.bincount(selected_array, minlength=len(expert_names))
        results["expert_usage_rows"] = [
            {
                "expert": expert_name,
                "usage": float(expert_counts[index] / max(selected_array.size, 1)),
                "count": int(expert_counts[index]),
                "num_samples": int(selected_array.size),
            }
            for index, expert_name in enumerate(expert_names)
        ]

        snr_binned_expert_rows: list[dict[str, Any]] = []
        if snr_bins is not None and snr_bins > 0 and "snr_binned_rows" in results:
            for snr_row in results["snr_binned_rows"]:
                usage = snr_row.get("expert_usage")
                if not usage:
                    continue
                for expert_name, expert_usage in usage.items():
                    snr_binned_expert_rows.append(
                        {
                            "bin_index": snr_row["bin_index"],
                            "snr_min": snr_row["snr_min"],
                            "snr_max": snr_row["snr_max"],
                            "snr_center": snr_row["snr_center"],
                            "expert": expert_name,
                            "usage": float(expert_usage),
                            "num_samples": snr_row["num_samples"],
                        }
                    )
        if snr_binned_expert_rows:
            results["snr_binned_expert_rows"] = snr_binned_expert_rows

    return results


def _format_snr_bin_labels(all_results: dict[str, dict[str, Any]], max_bins: int) -> list[str]:
    labels_per_profile: list[list[str]] = []
    for results in all_results.values():
        rows = list(results.get("snr_binned_rows", []))
        if not rows:
            continue
        labels_per_profile.append([f"{row['snr_min']:.1f}-{row['snr_max']:.1f}" for row in rows[:max_bins]])

    if not labels_per_profile:
        return [str(index) for index in range(max_bins)]

    first = labels_per_profile[0]
    if all(labels == first for labels in labels_per_profile[1:]):
        return first
    return [f"bin {index}" for index in range(max_bins)]


def _build_snr_heatmap_image(all_results: dict[str, dict[str, Any]]) -> wandb.Image | None:
    profiles = [profile for profile, results in all_results.items() if results.get("snr_binned_rows")]
    if not profiles:
        return None

    max_bins = max(len(all_results[profile].get("snr_binned_rows", [])) for profile in profiles)
    ber_matrix = np.full((len(profiles), max_bins), np.nan, dtype=np.float64)
    bler_matrix = np.full((len(profiles), max_bins), np.nan, dtype=np.float64)

    for row_index, profile in enumerate(profiles):
        for snr_row in all_results[profile].get("snr_binned_rows", []):
            bin_index = int(snr_row["bin_index"])
            if 0 <= bin_index < max_bins:
                ber_matrix[row_index, bin_index] = float(snr_row["ber"])
                bler_matrix[row_index, bin_index] = float(snr_row["bler"])

    snr_labels = _format_snr_bin_labels({profile: all_results[profile] for profile in profiles}, max_bins)
    fig, axes = plt.subplots(1, 2, figsize=(max(8, max_bins * 1.2), 4 + len(profiles) * 0.4), constrained_layout=True)
    for axis, matrix, title in zip(axes, [ber_matrix, bler_matrix], ["BER", "BLER"]):
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
        axis.set_title(f"SNR-Binned {title}")
        axis.set_xlabel("SNR bin")
        axis.set_ylabel("Profile")
        axis.set_xticks(range(max_bins))
        axis.set_xticklabels(snr_labels, rotation=45, ha="right")
        axis.set_yticks(range(len(profiles)))
        axis.set_yticklabels(profiles)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    image = wandb.Image(fig)
    plt.close(fig)
    return image


def _build_failure_grid_heatmap_image(all_results: dict[str, dict[str, Any]]) -> wandb.Image | None:
    profiles = [profile for profile, results in all_results.items() if results.get("failure_grid") is not None]
    if not profiles:
        return None

    fig, axes = plt.subplots(
        1,
        len(profiles),
        figsize=(6 * len(profiles), 6),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes[0]

    for axis, profile in zip(axes_flat, profiles):
        grid = np.asarray(all_results[profile]["failure_grid"], dtype=np.float64)
        failed_block_count = int(all_results[profile]["failure_grid_rows"][0]["failed_block_count"])
        image = axis.imshow(grid, aspect="auto", interpolation="nearest", cmap="magma", origin="lower")
        axis.set_title(f"{profile} failed-block symbol error rate\nfailed blocks={failed_block_count}")
        axis.set_xlabel("OFDM symbol")
        axis.set_ylabel("Subcarrier")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    image = wandb.Image(fig)
    plt.close(fig)
    return image


def print_results(results: dict[str, Any], dataset_name: str) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"Results for: {dataset_name}")
    print(f"{'=' * 60}")
    print(f"  Samples:      {results['num_samples']:,}")
    print(f"  Loss total:   {results['loss_total']:.6f}")
    print(f"  Loss BCE:     {results['loss_bce']:.6f}")
    print(f"  BER:          {results['ber']:.6e}")
    print(f"  BLER:         {results['bler']:.4f} ({results['num_block_errors']:,} / {results['num_samples']:,})")
    print(f"  SER:          {results['ser']:.6e}")
    print(f"  Channel MSE:  {results['channel_mse']:.6f}")
    if "realized_flops" in results:
        print(f"  Realized FLOPs: {results['realized_flops']:.3e}")
    if "expected_flops" in results:
        print(f"  Expected FLOPs: {results['expected_flops']:.3e}")

    if "snr_binned" in results:
        print("\n  SNR-Binned BLER:")
        print(f"  {'SNR (dB)':>12} | {'BLER':>10} | {'Samples':>10}")
        print(f"  {'-' * 12}-+-{'-' * 10}-+-{'-' * 10}")
        for bin_data in results["snr_binned"].values():
            snr_range = bin_data["snr_range"]
            print(
                f"  {snr_range[0]:5.1f}-{snr_range[1]:5.1f} | {bin_data['bler']:10.4f} | {bin_data['num_samples']:10,}"
            )


def _resolve_datasets(eval_cfg: DictConfig, checkpoint_cfg: dict[str, Any]) -> list[tuple[str, Path]]:
    profile_list = eval_cfg.get("profiles")
    if profile_list:
        data_dir = Path(to_absolute_path(str(eval_cfg.get("data_dir", PROJECT_ROOT / "data" / "test"))))
        return [(str(profile), data_dir / f"{profile}.pt") for profile in profile_list]

    if bool(eval_cfg.get("all_profiles", False)):
        profiles = ["uma", "tdlc", "mixed"]
        data_dir = Path(to_absolute_path(str(eval_cfg.get("data_dir", PROJECT_ROOT / "data" / "test"))))
        return [(profile, data_dir / f"{profile}.pt") for profile in profiles]

    default_dataset = eval_cfg.get("dataset_path")
    if default_dataset is None:
        default_dataset = checkpoint_cfg.get("evaluation", {}).get("dataset_path", "data/test/mixed.pt")
    default_path = Path(to_absolute_path(str(default_dataset)))
    return [(default_path.stem, default_path)]


def _resolve_dataset_with_artifact(cfg: DictConfig, dataset_name: str, local_path: Path) -> Path:
    if wandb.run is None or not bool(cfg.evaluation.get("use_artifacts", False)):
        return local_path

    artifact_name = build_dataset_artifact_name("test", dataset_name)
    artifact_alias = str(cfg.evaluation.get("dataset_artifact_alias", "latest"))
    project = str(cfg.logging.get("project") or cfg.project.name)
    entity = cfg.logging.get("entity")
    artifact_ref = f"{project}/{artifact_name}:{artifact_alias}"
    if entity:
        artifact_ref = f"{entity}/{artifact_ref}"

    try:
        artifact_path = use_artifact_path(artifact_ref, expected_filename=local_path.name)
        wandb.run.summary[f"artifacts/dataset/{dataset_name}"] = artifact_ref
        print(f"[INFO] Using dataset artifact for {dataset_name}: {artifact_ref}")
        return artifact_path
    except Exception as exc:
        print(f"[WARNING] Unable to use dataset artifact {artifact_ref}: {exc}")
        return local_path


def _init_wandb_eval_run(
    cfg: DictConfig,
    checkpoint_cfg: dict[str, Any],
    checkpoint_path: Path,
    dataset_names: list[str],
) -> bool:
    if not bool(cfg.logging.use_wandb) or not bool(cfg.evaluation.get("log_to_wandb", False)):
        return False

    checkpoint_logging = checkpoint_cfg.get("logging", {})
    checkpoint_experiment = checkpoint_cfg.get("experiment", {})
    checkpoint_model = checkpoint_cfg.get("model", {})

    project_name = str(
        cfg.logging.get("project")
        or checkpoint_logging.get("project")
        or checkpoint_cfg.get("project", {}).get("name", "moe-5g-nrx")
    )
    entity = cfg.logging.get("entity") or checkpoint_logging.get("entity")
    batch_name = checkpoint_experiment.get("batch_name")
    exp_name = checkpoint_experiment.get("exp_name") or checkpoint_path.stem
    dataset_suffix = "-".join(dataset_names)
    run_name = f"{exp_name}_eval_{dataset_suffix}"

    tags = [str(checkpoint_model.get("family", "static_dense")), "eval"]
    if batch_name:
        tags.append(str(batch_name).replace("-", "_"))
    exp_tags = checkpoint_experiment.get("tags")
    if exp_tags:
        tags.extend([str(tag) for tag in exp_tags])
    tags = list(dict.fromkeys(tags))

    config_payload = {
        "registry": {
            "run_role": "evaluation",
            "study_slug": checkpoint_experiment.get("study_slug"),
            "study_path": checkpoint_experiment.get("study_path"),
            "question": checkpoint_experiment.get("question"),
            "batch_name": batch_name,
            "exp_name": exp_name,
        },
        "evaluation": {
            "checkpoint": str(checkpoint_path),
            "profiles": dataset_names,
            "batch_size": int(cfg.evaluation.batch_size),
            "max_samples": cfg.evaluation.max_samples,
            "snr_bins": cfg.evaluation.snr_bins,
            "device": str(cfg.evaluation.device),
        },
        "checkpoint": checkpoint_cfg,
    }

    wandb.init(
        project=project_name,
        entity=str(entity) if entity else None,
        name=run_name,
        group=batch_name,
        tags=tags,
        job_type="evaluation",
        config=config_payload,
    )
    if wandb.run is not None:
        wandb.run.summary["registry/run_role"] = "evaluation"
        wandb.run.summary["registry/batch_name"] = batch_name
        wandb.run.summary["registry/exp_name"] = exp_name
        wandb.run.summary["registry/study_slug"] = checkpoint_experiment.get("study_slug")
        wandb.run.summary["registry/study_path"] = checkpoint_experiment.get("study_path")
        wandb.run.summary["registry/question"] = checkpoint_experiment.get("question")
    return True


def _link_checkpoint_artifact(checkpoint: dict[str, Any], explicit_artifact_ref: str | None) -> None:
    if wandb.run is None:
        return

    artifact_ref = explicit_artifact_ref
    if artifact_ref is None:
        wandb_info = checkpoint.get("wandb") if isinstance(checkpoint, dict) else None
        if isinstance(wandb_info, dict):
            artifact_ref = wandb_info.get("checkpoint_artifact")

    if not artifact_ref:
        return

    try:
        wandb.run.use_artifact(str(artifact_ref))
        wandb.run.summary["artifacts/source_checkpoint"] = str(artifact_ref)
    except Exception as exc:
        print(f"[WARNING] Unable to link checkpoint artifact {artifact_ref}: {exc}")


def _log_results_to_wandb(
    all_results: dict[str, dict[str, Any]],
    *,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_artifact: str | None,
) -> None:
    if wandb.run is None:
        return

    scalar_metrics: dict[str, Any] = {}
    checkpoint_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    experiment_cfg = checkpoint_cfg.get("experiment", {}) if isinstance(checkpoint_cfg, dict) else {}
    source_checkpoint = checkpoint_artifact or wandb.run.summary.get("artifacts/source_checkpoint")
    wandb_info = checkpoint.get("wandb") if isinstance(checkpoint, dict) else {}

    comparison_table = wandb.Table(
        columns=[
            "profile",
            "study_slug",
            "study_path",
            "batch_name",
            "exp_name",
            "source_train_run_path",
            "checkpoint_path",
            "checkpoint_artifact",
            "global_step",
            "num_parameters",
            "loss_total",
            "loss_bce",
            "ber",
            "bler",
            "ser",
            "channel_mse",
            "expected_flops",
            "realized_flops",
            "num_samples",
            "dataset_artifact",
        ]
    )
    snr_table = wandb.Table(
        columns=["profile", "bin_index", "snr_min", "snr_max", "snr_center", "ber", "bler", "num_samples"]
    )
    failure_table = wandb.Table(
        columns=["profile", "sample_index", "snr_db", "bit_errors", "bits_per_sample", "ber", "block_error"]
    )
    failure_grid_table = wandb.Table(
        columns=["profile", "subcarrier_index", "ofdm_symbol_index", "symbol_error_rate", "failed_block_count"]
    )
    expert_usage_table = wandb.Table(columns=["profile", "expert", "usage", "count", "num_samples"])
    expert_snr_table = wandb.Table(
        columns=["profile", "bin_index", "snr_min", "snr_max", "snr_center", "expert", "usage", "num_samples"]
    )
    snr_row_count = 0
    failure_row_count = 0
    failure_grid_row_count = 0
    expert_usage_row_count = 0
    expert_snr_row_count = 0

    for profile, results in all_results.items():
        scalar_metrics[f"eval/{profile}/loss_total"] = results["loss_total"]
        scalar_metrics[f"eval/{profile}/loss_bce"] = results["loss_bce"]
        scalar_metrics[f"eval/{profile}/ber"] = results["ber"]
        scalar_metrics[f"eval/{profile}/bler"] = results["bler"]
        scalar_metrics[f"eval/{profile}/ser"] = results["ser"]
        scalar_metrics[f"eval/{profile}/channel_mse"] = results["channel_mse"]
        scalar_metrics[f"eval/{profile}/num_samples"] = results["num_samples"]
        if "expected_flops" in results:
            scalar_metrics[f"eval/{profile}/expected_flops"] = results["expected_flops"]
        if "realized_flops" in results:
            scalar_metrics[f"eval/{profile}/realized_flops"] = results["realized_flops"]
        dataset_artifact = wandb.run.summary.get(f"artifacts/dataset/{profile}")
        comparison_table.add_data(
            profile,
            experiment_cfg.get("study_slug"),
            experiment_cfg.get("study_path"),
            experiment_cfg.get("batch_name"),
            experiment_cfg.get("exp_name"),
            wandb_info.get("run_path") if isinstance(wandb_info, dict) else None,
            str(checkpoint_path),
            source_checkpoint,
            checkpoint.get("global_step"),
            wandb.run.summary.get("model/num_parameters"),
            results["loss_total"],
            results["loss_bce"],
            results["ber"],
            results["bler"],
            results["ser"],
            results["channel_mse"],
            results.get("expected_flops"),
            results.get("realized_flops"),
            results["num_samples"],
            dataset_artifact,
        )

        for snr_row in results.get("snr_binned_rows", []):
            snr_table.add_data(
                profile,
                snr_row["bin_index"],
                snr_row["snr_min"],
                snr_row["snr_max"],
                snr_row["snr_center"],
                snr_row["ber"],
                snr_row["bler"],
                snr_row["num_samples"],
            )
            snr_row_count += 1

        for failure_row in results.get("failure_examples", []):
            failure_table.add_data(
                profile,
                failure_row["sample_index"],
                failure_row["snr_db"],
                failure_row["bit_errors"],
                failure_row["bits_per_sample"],
                failure_row["ber"],
                failure_row["block_error"],
            )
            failure_row_count += 1

        for failure_grid_row in results.get("failure_grid_rows", []):
            failure_grid_table.add_data(
                profile,
                failure_grid_row["subcarrier_index"],
                failure_grid_row["ofdm_symbol_index"],
                failure_grid_row["symbol_error_rate"],
                failure_grid_row["failed_block_count"],
            )
            failure_grid_row_count += 1

        for expert_usage_row in results.get("expert_usage_rows", []):
            expert_usage_table.add_data(
                profile,
                expert_usage_row["expert"],
                expert_usage_row["usage"],
                expert_usage_row["count"],
                expert_usage_row["num_samples"],
            )
            scalar_metrics[f"eval/{profile}/expert_usage/{expert_usage_row['expert']}"] = expert_usage_row["usage"]
            expert_usage_row_count += 1

        for expert_snr_row in results.get("snr_binned_expert_rows", []):
            expert_snr_table.add_data(
                profile,
                expert_snr_row["bin_index"],
                expert_snr_row["snr_min"],
                expert_snr_row["snr_max"],
                expert_snr_row["snr_center"],
                expert_snr_row["expert"],
                expert_snr_row["usage"],
                expert_snr_row["num_samples"],
            )
            expert_snr_row_count += 1

    wandb.log(scalar_metrics)
    tables_to_log: dict[str, Any] = {
        "eval/comparison": comparison_table,
        "eval/summary": comparison_table,
    }
    if snr_row_count > 0:
        tables_to_log["eval/snr_binned"] = snr_table
    if failure_row_count > 0:
        tables_to_log["eval/failures"] = failure_table
    if failure_grid_row_count > 0:
        tables_to_log["eval/failure_grid"] = failure_grid_table
    if expert_usage_row_count > 0:
        tables_to_log["eval/expert_usage"] = expert_usage_table
    if expert_snr_row_count > 0:
        tables_to_log["eval/expert_usage_snr_binned"] = expert_snr_table

    snr_heatmap = _build_snr_heatmap_image(all_results)
    if snr_heatmap is not None:
        tables_to_log["viz/snr_binned_error_heatmap"] = snr_heatmap

    failure_grid_heatmap = _build_failure_grid_heatmap_image(all_results)
    if failure_grid_heatmap is not None:
        tables_to_log["viz/failure_grid_heatmap"] = failure_grid_heatmap

    wandb.log(tables_to_log)

    for key, value in scalar_metrics.items():
        wandb.run.summary[key] = value


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    eval_cfg = cfg.evaluation
    checkpoint_artifact = eval_cfg.get("checkpoint_artifact")
    checkpoint_raw = eval_cfg.get("checkpoint")
    if checkpoint_artifact is None and checkpoint_raw is None:
        print("[ERROR] Missing evaluation.checkpoint in Hydra config")
        sys.exit(1)

    if checkpoint_artifact is not None:
        checkpoint_path = _download_checkpoint_artifact(str(checkpoint_artifact))
    else:
        checkpoint_path = Path(to_absolute_path(str(checkpoint_raw)))
    requested_device = str(eval_cfg.get("device", cfg.runtime.device))

    # Resolve device
    device = torch.device(requested_device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, config = load_checkpoint(checkpoint_path, device)
    bits_per_symbol = int(config["data"]["bits_per_symbol"])
    channel_loss_weight = float(config["model"]["compute"]["channel_loss_weight"])
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"  Model: {config['model']['name']}")
    print(f"  Global step: {checkpoint['global_step']}")

    # Determine datasets to evaluate
    datasets = _resolve_datasets(eval_cfg, config)
    _init_wandb_eval_run(cfg, config, checkpoint_path, [name for name, _ in datasets])
    _link_checkpoint_artifact(checkpoint, str(checkpoint_artifact) if checkpoint_artifact is not None else None)
    if wandb.run is not None:
        wandb_info = checkpoint.get("wandb") if isinstance(checkpoint, dict) else None
        if isinstance(wandb_info, dict):
            wandb.run.summary["registry/source_train_run_path"] = wandb_info.get("run_path")
        wandb.run.summary["model/num_parameters"] = num_parameters
        wandb.run.summary["model/global_step"] = checkpoint["global_step"]
        wandb.run.summary["model/name"] = str(config["model"]["name"])
        wandb.run.summary["model/family"] = str(config["model"]["family"])

    # Evaluate each dataset
    all_results = {}
    for name, local_path in datasets:
        path = _resolve_dataset_with_artifact(cfg, name, local_path)
        if not path.exists():
            print(f"\n[WARNING] Dataset not found: {path}, skipping")
            continue

        print(f"\nLoading dataset: {path}")
        dataloader = build_cached_dataloader(
            path=path,
            batch_size=int(eval_cfg.batch_size),
            max_samples=int(eval_cfg.max_samples) if eval_cfg.max_samples is not None else None,
            shuffle=False,
        )

        results = evaluate_dataset(
            model=model,
            dataloader=dataloader,
            device=device,
            bits_per_symbol=bits_per_symbol,
            channel_loss_weight=channel_loss_weight,
            snr_bins=int(eval_cfg.snr_bins) if eval_cfg.snr_bins is not None else None,
            failure_examples_limit=int(eval_cfg.get("failure_examples_limit", 20)),
        )
        all_results[name] = results
        print_results(results, name)

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("Summary")
        print(f"{'=' * 60}")
        print(f"  {'Profile':>10} | {'BLER':>10} | {'BER':>12} | {'Loss':>10}")
        print(f"  {'-' * 10}-+-{'-' * 10}-+-{'-' * 12}-+-{'-' * 10}")
        for name, results in all_results.items():
            print(f"  {name:>10} | {results['bler']:10.4f} | {results['ber']:12.6e} | {results['loss_total']:10.6f}")

    _log_results_to_wandb(
        all_results,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        checkpoint_artifact=str(checkpoint_artifact) if checkpoint_artifact is not None else None,
    )
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
