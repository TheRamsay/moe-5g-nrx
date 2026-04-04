#!/usr/bin/env python


from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
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
from src.models import StaticDenseNRX  # noqa: E402
from src.utils.metrics import compute_batch_error_metrics  # noqa: E402
from src.utils.wandb_artifacts import build_dataset_artifact_name, use_artifact_path  # noqa: E402


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Build model from saved config
    model = StaticDenseNRX(
        input_channels=int(config["data"]["input_channels"]),
        output_bits=int(config["data"]["bits_per_symbol"])
        * int(config["data"]["num_subcarriers"])
        * int(config["data"]["num_ofdm_symbols"]),
        state_dim=int(config["model"]["backbone"]["state_dim"]),
        num_cnn_blocks=int(config["model"]["backbone"]["num_cnn_blocks"]),
        stem_hidden_dims=list(config["model"]["backbone"]["stem_hidden_dims"]),
        block_hidden_dim=int(config["model"]["backbone"]["block_hidden_dim"]),
        readout_hidden_dim=int(config["model"]["backbone"]["readout_hidden_dim"]),
        activation=str(config["model"]["backbone"]["activation"]),
        num_subcarriers=int(config["data"]["num_subcarriers"]),
        num_ofdm_symbols=int(config["data"]["num_ofdm_symbols"]),
        bits_per_symbol=int(config["data"]["bits_per_symbol"]),
        num_rx_antennas=int(config["data"]["num_rx_antennas"]),
        pilot_symbol_indices=list(config["model"]["backbone"]["pilot_symbol_indices"]),
        pilot_subcarrier_spacing=int(config["model"]["backbone"]["pilot_subcarrier_spacing"]),
    )
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
    sample_offset = 0

    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
        batch: CachedNRXBatch
        features = batch.inputs.to(device)
        targets = batch.bit_labels.flatten(start_dim=1).to(device)
        channel_targets = batch.channel_target.to(device)
        snr_db = batch.snr_db

        outputs = model(features)
        logits = outputs["logits"]
        channel_estimate = outputs["channel_estimate"]

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

        # Per-sample bit errors (for SNR analysis)
        per_sample_bit_errors = bit_errors.sum(dim=1).cpu().numpy()
        per_sample_bits = targets.shape[1]
        per_sample_block_error = error_metrics.block_errors.cpu().numpy().astype(int)
        sample_ber = per_sample_bit_errors / max(per_sample_bits, 1)

        all_snr_db.extend(snr_db.numpy().tolist())
        all_bit_errors.extend(per_sample_bit_errors.tolist())
        all_bits_per_sample.extend([per_sample_bits] * len(snr_db))
        all_block_errors.extend(per_sample_block_error.tolist())

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

        results["snr_binned"] = snr_binned
        results["snr_binned_rows"] = snr_binned_rows

    if failure_examples_limit > 0:
        failure_examples_sorted = sorted(
            failure_examples,
            key=lambda row: (row["bit_errors"], row["block_error"], -row["snr_db"]),
            reverse=True,
        )
        results["failure_examples"] = failure_examples_sorted[:failure_examples_limit]

    return results


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
    snr_row_count = 0
    failure_row_count = 0

    for profile, results in all_results.items():
        scalar_metrics[f"eval/{profile}/loss_total"] = results["loss_total"]
        scalar_metrics[f"eval/{profile}/loss_bce"] = results["loss_bce"]
        scalar_metrics[f"eval/{profile}/ber"] = results["ber"]
        scalar_metrics[f"eval/{profile}/bler"] = results["bler"]
        scalar_metrics[f"eval/{profile}/ser"] = results["ser"]
        scalar_metrics[f"eval/{profile}/channel_mse"] = results["channel_mse"]
        scalar_metrics[f"eval/{profile}/num_samples"] = results["num_samples"]
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

    wandb.log(scalar_metrics)
    tables_to_log: dict[str, Any] = {
        "eval/comparison": comparison_table,
        "eval/summary": comparison_table,
    }
    if snr_row_count > 0:
        tables_to_log["eval/snr_binned"] = snr_table
    if failure_row_count > 0:
        tables_to_log["eval/failures"] = failure_table
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
