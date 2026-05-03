"""PCA-2D overlay of in-distribution (UMa+TDL-C) vs OOD (ASU ray-traced) stem features.

Loads exp26, captures pooled stem features for the three profiles, fits a 2-D
PCA on the in-distribution features, projects ASU OOD samples onto the same
basis, and plots the overlay. Visual evidence that OOD samples land outside
the in-distribution feature manifold, explaining why the model fails on ASU.

Output: docs/figures/pca_ood_overlay.{pdf,png}
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


def _download_artifact(ref: str) -> Path:
    import wandb
    api = wandb.Api()
    art = api.artifact(ref, type="model")
    art_dir = Path(art.download(root=f"/tmp/wandb_oodpca/{ref.replace('/', '_').replace(':', '_')}"))
    for path in art_dir.rglob("*.pt"):
        return path
    raise FileNotFoundError(f"No .pt in {art_dir}")


def collect_stem_features(model, loader, device, max_samples):
    model.eval()
    captured = {}
    def hook(_m, _i, output):
        captured["s"] = output
    handle = model.stem.register_forward_hook(hook)
    feats = []
    n = 0
    with torch.inference_mode():
        for batch in loader:
            if n >= max_samples:
                break
            x = batch.inputs.to(device, non_blocking=True)
            _ = model(x)
            shared = captured["s"]
            pooled = torch.cat([shared.mean(dim=(2, 3)), shared.amax(dim=(2, 3))], dim=1)
            feats.append(pooled.cpu().numpy())
            n += x.shape[0]
    handle.remove()
    return np.concatenate(feats, axis=0)[:max_samples]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="knn_moe-5g-nrx/moe-5g-nrx/model-moe_alphasweep_asym_a2e3_s67-t6lkdep2:best")
    ap.add_argument("--uma-path", type=Path, default=Path("/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test/uma.pt"))
    ap.add_argument("--tdlc-path", type=Path, default=Path("/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test/tdlc.pt"))
    ap.add_argument("--asu-path", type=Path, default=Path("/storage/brno2/home/ramsay/moe-5g-datasets/deepmimo-train/asu_campus1/asu_campus1"))
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] downloading checkpoint {args.checkpoint}", file=sys.stderr)
    ckpt_path = _download_artifact(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model_from_config(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    print(f"[INFO] collecting stem features (max {args.max_samples} per profile)", file=sys.stderr)
    feats = {}
    for name, path in [("uma", args.uma_path), ("tdlc", args.tdlc_path), ("asu", args.asu_path)]:
        loader = build_cached_dataloader(path=path, batch_size=256, num_workers=0)
        feats[name] = collect_stem_features(model, loader, device, args.max_samples)
        print(f"[INFO] {name}: {feats[name].shape}", file=sys.stderr)

    # Fit PCA on in-distribution (UMa + TDLC), project all 3 onto same basis.
    in_dist = np.concatenate([feats["uma"], feats["tdlc"]], axis=0)
    mu = in_dist.mean(axis=0, keepdims=True)
    centered = in_dist - mu
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T  # (D, 2)

    proj = {name: (f - mu) @ basis for name, f in feats.items()}

    # Save numpy bundle for later re-plotting if needed.
    np.savez(args.out / "pca_ood_overlay.npz",
             uma_proj=proj["uma"], tdlc_proj=proj["tdlc"], asu_proj=proj["asu"],
             basis=basis, mu=mu)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(proj["uma"][:, 0],  proj["uma"][:, 1],  s=6, alpha=0.45, color="#56B4E9", label=f"UMa (in-dist, n={len(proj['uma'])})")
    ax.scatter(proj["tdlc"][:, 0], proj["tdlc"][:, 1], s=6, alpha=0.45, color="#0072B2", label=f"TDL-C (in-dist, n={len(proj['tdlc'])})")
    ax.scatter(proj["asu"][:, 0],  proj["asu"][:, 1],  s=8, alpha=0.55, color="#D55E00", marker="x", label=f"ASU ray-traced (OOD, n={len(proj['asu'])})")

    ax.set_xlabel("PC 1 (in-distribution basis)")
    ax.set_ylabel("PC 2 (in-distribution basis)")
    ax.set_title("Stem feature space: in-distribution vs OOD")
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "pca_ood_overlay.pdf", bbox_inches="tight")
    fig.savefig(args.out / "pca_ood_overlay.png", dpi=160, bbox_inches="tight")
    print(f"[INFO] wrote {args.out}/pca_ood_overlay.{{pdf,png}} and pca_ood_overlay.npz", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
