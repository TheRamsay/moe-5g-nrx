"""t-SNE visualization of MoE router channel features colored by expert choice.

Loads the exp26 checkpoint, runs forward on a sample of test data (uma+tdlc),
extracts the pooled stem features (router input), gets the router's selected
expert per sample, and plots a 2-D t-SNE colored by chosen expert.

Shows whether channel-quality features cluster by routing decision — i.e.
whether the router has learned a meaningful partition of the channel space.

Usage: `uv run python scripts/visualize_channel_features.py --device cuda --out figures/`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_cached_dataloader  # noqa: E402
from src.models import build_model_from_config  # noqa: E402


def pca_2d(features: np.ndarray) -> np.ndarray:
    """Numpy SVD-based 2-D PCA. Faster + zero deps vs scikit-learn TSNE."""
    centered = features - features.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


EXPERT_COLORS = {0: "#4daf4a", 1: "#377eb8", 2: "#e41a1c"}  # nano / small / large
EXPERT_NAMES = {0: "nano", 1: "small", 2: "large"}


def _download_artifact(ref: str) -> Path:
    import wandb

    api = wandb.Api()
    art = api.artifact(ref, type="model")
    art_dir = Path(art.download(root=f"/tmp/wandb_tsne/{ref.replace('/', '_').replace(':', '_')}"))
    for path in art_dir.rglob("*.pt"):
        return path
    raise FileNotFoundError(f"No .pt in {art_dir}")


def collect_features(
    model: torch.nn.Module, loader, device: torch.device, max_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run forward, capture pooled stem features + selected expert + true SNR."""
    model.eval()
    captured = {}

    def hook(_module, _input, output):
        captured["shared_state"] = output

    handle = model.stem.register_forward_hook(hook)
    feats: list[np.ndarray] = []
    experts: list[np.ndarray] = []
    snrs: list[np.ndarray] = []
    n_collected = 0
    with torch.inference_mode():
        for batch in loader:
            if n_collected >= max_samples:
                break
            x = batch.inputs.to(device, non_blocking=True)
            out = model(x)
            shared = captured["shared_state"]
            pooled = torch.cat([shared.mean(dim=(2, 3)), shared.amax(dim=(2, 3))], dim=1)
            feats.append(pooled.cpu().numpy())
            experts.append(out["selected_expert_index"].cpu().numpy())
            snrs.append(batch.snr_db.cpu().numpy())
            n_collected += x.shape[0]
    handle.remove()
    feats_arr = np.concatenate(feats, axis=0)[:max_samples]
    experts_arr = np.concatenate(experts, axis=0)[:max_samples]
    snrs_arr = np.concatenate(snrs, axis=0)[:max_samples]
    return feats_arr, experts_arr, snrs_arr


def plot_tsne(features: np.ndarray, experts: np.ndarray, snrs: np.ndarray, profile: str, out_path: Path) -> None:
    print(f"[INFO] {profile}: PCA-2D on {len(features)} samples, dim={features.shape[1]}", file=sys.stderr)
    embed = pca_2d(features)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # Left: colored by chosen expert
    for e in (0, 1, 2):
        mask = experts == e
        if mask.sum() == 0:
            continue
        axes[0].scatter(
            embed[mask, 0],
            embed[mask, 1],
            s=4,
            alpha=0.5,
            c=EXPERT_COLORS[e],
            label=f"{EXPERT_NAMES[e]} ({mask.sum()})",
        )
    axes[0].set_title(f"{profile.upper()}: stem features by router choice")
    axes[0].legend(loc="best", title="Expert")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    # Right: colored by true SNR
    sc = axes[1].scatter(embed[:, 0], embed[:, 1], s=4, alpha=0.6, c=snrs, cmap="viridis")
    axes[1].set_title(f"{profile.upper()}: stem features by true SNR")
    fig.colorbar(sc, ax=axes[1], label="SNR (dB)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[INFO] wrote {out_path}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="knn_moe-5g-nrx/moe-5g-nrx/model-moe_alphasweep_asym_a2e3_s67-t6lkdep2:best"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test")
    )
    parser.add_argument("--profiles", nargs="+", default=["uma", "tdlc"])
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = _download_artifact(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model_from_config(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    for profile in args.profiles:
        path = args.data_dir / f"{profile}.pt"
        if not path.exists():
            print(f"[WARNING] no test file for {profile} at {path}, skipping", file=sys.stderr)
            continue
        loader = build_cached_dataloader(path=path, batch_size=256, num_workers=0)
        feats, experts, snrs = collect_features(model, loader, device, args.max_samples)
        out_png = args.out / f"channel_feature_tsne_{profile}.png"
        plot_tsne(feats, experts, snrs, profile, out_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
