"""Wall-clock latency benchmark for dense baselines + exp26 MoE.

Measures forward-pass latency (samples/sec) for each model on CPU and GPU
at realistic batch sizes. Translates the FLOPs claim into deployment-relevant
inference speedup.

Usage: `uv run python scripts/benchmark_latency.py --device cuda --out latency.json`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model_from_config  # noqa: E402


def load_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = build_model_from_config(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, config


# (label, W&B artifact ref). Local checkpoints are downloaded on demand.
CHECKPOINTS = [
    ("dense_nano", "knn_moe-5g-nrx/moe-5g-nrx/model-dense_nano_final20k_constant_lr_s67-aos4hhid:best"),
    ("dense_small", "knn_moe-5g-nrx/moe-5g-nrx/model-dense_small_final20k_constant_lr_s67-kivdz4qu:best"),
    ("dense_large", "knn_moe-5g-nrx/moe-5g-nrx/model-dense_large_final20k_constant_lr_s67-55l1dpby:best"),
    ("exp26_moe", "knn_moe-5g-nrx/moe-5g-nrx/model-moe_alphasweep_asym_a2e3_s67-t6lkdep2:best"),
]


def _download_artifact(ref: str) -> Path:
    import wandb

    api = wandb.Api()
    art = api.artifact(ref, type="model")
    art_dir = Path(art.download(root=f"/tmp/wandb_latency/{ref.replace('/', '_').replace(':', '_')}"))
    for path in art_dir.rglob("*.pt"):
        return path
    raise FileNotFoundError(f"No .pt file in {art_dir}")


def benchmark_model_synthetic(
    model: torch.nn.Module, device: torch.device, batch_size: int, n_warmup: int = 20, n_iter: int = 100
) -> dict:
    """Time `n_iter` forward passes on random Gaussian input."""
    model.eval()
    model.to(device)

    in_channels = getattr(model, "input_channels", 16)
    x = torch.randn(batch_size, in_channels, 128, 14, device=device)

    with torch.inference_mode():
        for _ in range(n_warmup):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(n_iter):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "ms_per_batch": (elapsed / n_iter) * 1000,
        "samples_per_s": batch_size * n_iter / elapsed,
        "batch_size": batch_size,
        "n_iter": n_iter,
        "data": "synthetic_gaussian",
    }


def benchmark_model_real(
    model: torch.nn.Module,
    device: torch.device,
    data_path: Path,
    batch_size: int,
    n_warmup: int = 20,
    n_iter: int = 200,
) -> dict:
    """Time `n_iter` forward passes on real test data (cycles through batches)."""
    from src.data import build_cached_dataloader

    model.eval()
    model.to(device)
    loader = build_cached_dataloader(path=data_path, batch_size=batch_size, num_workers=0)
    batches = []
    for batch in loader:
        batches.append(batch.inputs.to(device, non_blocking=True))
        if len(batches) >= 8:  # cache 8 batches; cycle for n_iter
            break

    with torch.inference_mode():
        for i in range(n_warmup):
            _ = model(batches[i % len(batches)])
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(n_iter):
            _ = model(batches[i % len(batches)])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "ms_per_batch": (elapsed / n_iter) * 1000,
        "samples_per_s": batch_size * n_iter / elapsed,
        "batch_size": batch_size,
        "n_iter": n_iter,
        "data": str(data_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="If set, also benchmark on real test data from data_dir/{uma,tdlc}.pt",
    )
    parser.add_argument("--out", type=Path, default=Path("latency_results.json"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="If set, only benchmark these model labels (e.g. --models exp26_moe dense_large)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available; falling back to CPU", file=sys.stderr)
        device = torch.device("cpu")

    results = {"device": str(device), "batch_size": args.batch_size, "models": {}}
    real_profiles = []
    if args.data_dir is not None:
        for prof in ("uma", "tdlc"):
            if (args.data_dir / f"{prof}.pt").exists():
                real_profiles.append(prof)

    checkpoints = [(label, ref) for label, ref in CHECKPOINTS if args.models is None or label in args.models]
    for label, ref in checkpoints:
        print(f"[INFO] {label} ({ref})", file=sys.stderr)
        ckpt_path = _download_artifact(ref)
        model, _config = load_checkpoint(ckpt_path, device)
        n_params = sum(p.numel() for p in model.parameters())

        synth = benchmark_model_synthetic(model, device, args.batch_size, args.n_warmup, args.n_iter)
        per_profile = {}
        for prof in real_profiles:
            real = benchmark_model_real(
                model, device, args.data_dir / f"{prof}.pt", args.batch_size, args.n_warmup, args.n_iter
            )
            per_profile[prof] = real

        results["models"][label] = {
            "checkpoint": ref,
            "num_parameters": n_params,
            "synthetic": synth,
            "real": per_profile,
        }
        line = (
            f"  {label}: synth {synth['ms_per_batch']:.2f}ms"
            + "".join(f" | {p} {per_profile[p]['ms_per_batch']:.2f}ms" for p in real_profiles)
            + f" | {n_params / 1e3:.0f}k params"
        )
        print(line, file=sys.stderr)

    args.out.write_text(json.dumps(results, indent=2))
    print(f"[INFO] wrote {args.out}", file=sys.stderr)

    # Markdown table
    print(f"\n## Wall-clock latency on `{device}` (batch_size={args.batch_size}, n_iter={args.n_iter})\n")
    headers = ["Model", "Params"] + [f"{src} ms/batch" for src in (["synth"] + real_profiles)]
    headers += ["Speedup vs dense_large (synth)"]
    print("| " + " | ".join(headers) + " |")
    print("|---|---:|" + "---:|" * (len(headers) - 2) + "---:|")
    base = results["models"].get("dense_large", {}).get("synthetic", {}).get("ms_per_batch", float("nan"))
    for label, m in results["models"].items():
        synth_ms = m["synthetic"]["ms_per_batch"]
        row = [label, f"{m['num_parameters'] / 1e3:.0f}k", f"{synth_ms:.2f}"]
        for prof in real_profiles:
            row.append(f"{m['real'][prof]['ms_per_batch']:.2f}")
        speedup = base / synth_ms if base and synth_ms else float("nan")
        row.append(f"{speedup:.2f}×")
        print("| " + " | ".join(row) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
