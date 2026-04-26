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


def benchmark_model(
    model: torch.nn.Module, device: torch.device, batch_size: int, n_warmup: int = 20, n_iter: int = 100
) -> dict:
    """Time `n_iter` forward passes after `n_warmup`. Returns ms/batch + samples/sec."""
    model.eval()
    model.to(device)

    # Build a synthetic batch matching expected input shape per model:
    # [batch, channels, freq=128, time=14]; channels read from model attribute.
    in_channels = getattr(model, "input_channels", 16)
    x = torch.randn(batch_size, in_channels, 128, 14, device=device)

    # Warmup
    with torch.inference_mode():
        for _ in range(n_warmup):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Time
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(n_iter):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms_per_batch = (elapsed / n_iter) * 1000
    samples_per_s = batch_size * n_iter / elapsed
    return {
        "ms_per_batch": ms_per_batch,
        "samples_per_s": samples_per_s,
        "batch_size": batch_size,
        "n_iter": n_iter,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("latency_results.json"))
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available; falling back to CPU", file=sys.stderr)
        device = torch.device("cpu")

    results = {"device": str(device), "batch_size": args.batch_size, "models": {}}
    for label, ref in CHECKPOINTS:
        print(f"[INFO] {label} ({ref})", file=sys.stderr)
        ckpt_path = _download_artifact(ref)
        model, config = load_checkpoint(ckpt_path, device)
        n_params = sum(p.numel() for p in model.parameters())

        bench = benchmark_model(model, device, args.batch_size, args.n_warmup, args.n_iter)
        results["models"][label] = {
            "checkpoint": ref,
            "num_parameters": n_params,
            **bench,
        }
        print(
            f"  {label}: {bench['ms_per_batch']:.2f} ms/batch | "
            f"{bench['samples_per_s']:.1f} samples/sec | {n_params / 1e3:.0f}k params",
            file=sys.stderr,
        )

    args.out.write_text(json.dumps(results, indent=2))
    print(f"[INFO] wrote {args.out}", file=sys.stderr)

    # Print a markdown table to stdout
    print(f"\n## Wall-clock latency on `{device}` (batch_size={args.batch_size}, n_iter={args.n_iter})\n")
    print("| Model | Params | ms/batch | samples/sec | Speedup vs dense_large |")
    print("|---|---:|---:|---:|---:|")
    base = results["models"].get("dense_large", {}).get("ms_per_batch", float("nan"))
    for label, m in results["models"].items():
        speedup = base / m["ms_per_batch"] if base and m["ms_per_batch"] else float("nan")
        print(
            f"| {label} | {m['num_parameters'] / 1e3:.0f}k | "
            f"{m['ms_per_batch']:.2f} | {m['samples_per_s']:.1f} | {speedup:.2f}× |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
