"""Mechanistic analysis of the router: A) linear probing, C) per-expert
specialization, F) decision boundary on PCA plane.

Loads exp26 checkpoint, runs inference on UMa + TDLC test data, captures:
  - stem features (router input, 112-dim pooled)
  - selected_expert per sample
  - true SNR + channel target (for probing)
  - logits + true bits → per-sample BLER

Outputs three figure groups in docs/figures/:
  - router_mechanism_linear_probing.png + JSON of probe scores
  - router_mechanism_expert_specialization.png
  - router_mechanism_decision_boundary.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _linear_regression_r2(X_tr, y_tr, X_te, y_te) -> float:
    """Fit y = X @ w + b, return R² on test set. Pure numpy."""
    # add bias column
    Xb_tr = np.hstack([X_tr, np.ones((len(X_tr), 1))])
    Xb_te = np.hstack([X_te, np.ones((len(X_te), 1))])
    w, *_ = np.linalg.lstsq(Xb_tr, y_tr, rcond=None)
    pred = Xb_te @ w
    ss_res = ((y_te - pred) ** 2).sum()
    ss_tot = ((y_te - y_te.mean()) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def _binary_classification_acc(X_tr, y_tr, X_te, y_te) -> float:
    """Fit linear regression on {0,1} labels and threshold at 0.5 for accuracy."""
    Xb_tr = np.hstack([X_tr, np.ones((len(X_tr), 1))])
    Xb_te = np.hstack([X_te, np.ones((len(X_te), 1))])
    w, *_ = np.linalg.lstsq(Xb_tr, y_tr.astype(float), rcond=None)
    pred = (Xb_te @ w) > 0.5
    return float((pred == y_te).mean())


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_cached_dataloader  # noqa: E402
from src.models import build_model_from_config  # noqa: E402

EXPERT_NAMES = ["nano", "small", "large"]
EXPERT_COLORS = {"nano": "#fdc086", "small": "#7fc97f", "large": "#beaed4"}


def _download_artifact(ref: str) -> Path:
    import wandb

    api = wandb.Api()
    art = api.artifact(ref, type="model")
    art_dir = Path(art.download(root=f"/tmp/wandb_mechanism/{ref.replace('/', '_').replace(':', '_')}"))
    for path in art_dir.rglob("*.pt"):
        return path
    raise FileNotFoundError(f"No .pt file in {art_dir}")


def collect_data(model, loader, device, max_samples: int):
    """Run forward pass; capture stem features, routing, SNR, channel targets, BLER."""
    model.eval()
    captured = {}

    def hook(_module, _input, output):
        captured["shared_state"] = output

    handle = model.stem.register_forward_hook(hook)

    feats: list[np.ndarray] = []
    experts: list[np.ndarray] = []
    snrs: list[np.ndarray] = []
    chan_powers: list[np.ndarray] = []
    delay_spreads: list[np.ndarray] = []
    block_errors: list[np.ndarray] = []
    n_collected = 0

    with torch.inference_mode():
        for batch in loader:
            if n_collected >= max_samples:
                break
            x = batch.inputs.to(device, non_blocking=True)
            targets = batch.bit_labels.flatten(start_dim=1).to(device)
            channel_targets = batch.channel_target.to(device)
            out = model(x)

            shared = captured["shared_state"]
            pooled = torch.cat([shared.mean(dim=(2, 3)), shared.amax(dim=(2, 3))], dim=1)
            feats.append(pooled.cpu().numpy())
            experts.append(out["selected_expert_index"].cpu().numpy())
            snrs.append(batch.snr_db.cpu().numpy())

            # Channel power per sample (mean of |H|^2)
            n_ant = channel_targets.shape[1] // 2
            h_real = channel_targets[:, :n_ant]
            h_imag = channel_targets[:, n_ant:]
            cp = (h_real**2 + h_imag**2).mean(dim=(1, 2, 3)).cpu().numpy()
            chan_powers.append(cp)

            # Crude "delay spread" proxy: variance of channel power across freq dim
            #   (real channels have specific delay patterns; this captures spread)
            h_mag = (h_real**2 + h_imag**2).sum(dim=1).cpu().numpy()  # (B, freq, time)
            ds = h_mag.var(axis=1).mean(axis=1)
            delay_spreads.append(ds)

            # Per-sample block errors (any bit wrong)
            preds = (out["logits"] > 0).long()
            sample_block_err = (preds != targets.long()).any(dim=1).cpu().numpy().astype(int)
            block_errors.append(sample_block_err)

            n_collected += x.shape[0]

    handle.remove()
    return {
        "feats": np.concatenate(feats, axis=0)[:max_samples],
        "experts": np.concatenate(experts, axis=0)[:max_samples],
        "snr_db": np.concatenate(snrs, axis=0)[:max_samples],
        "chan_power": np.concatenate(chan_powers, axis=0)[:max_samples],
        "delay_spread": np.concatenate(delay_spreads, axis=0)[:max_samples],
        "block_err": np.concatenate(block_errors, axis=0)[:max_samples],
    }


def linear_probing(data_uma: dict, data_tdlc: dict, out_dir: Path) -> dict:
    """A: Train linear probes on stem features for physical channel parameters."""
    print("[A] Linear probing for physical features...", file=sys.stderr)

    X_uma, X_tdlc = data_uma["feats"], data_tdlc["feats"]
    X_all = np.concatenate([X_uma, X_tdlc], axis=0)
    profile_labels = np.concatenate([np.zeros(len(X_uma)), np.ones(len(X_tdlc))])

    results: dict[str, float] = {}

    # Profile classification (UMa vs TDLC) — should be near-perfect
    n = len(X_all)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    split = int(0.7 * n)
    tr, te = perm[:split], perm[split:]
    results["profile_acc"] = _binary_classification_acc(X_all[tr], profile_labels[tr], X_all[te], profile_labels[te])

    # Per-profile probes for SNR / channel_power / delay_spread (regression)
    for prof_name, d in [("uma", data_uma), ("tdlc", data_tdlc)]:
        X = d["feats"]
        n = len(X)
        perm = rng.permutation(n)
        split = int(0.7 * n)
        tr, te = perm[:split], perm[split:]
        for target_name, y in [
            ("snr", d["snr_db"]),
            ("chan_power", d["chan_power"]),
            ("delay_spread", d["delay_spread"]),
        ]:
            r2 = _linear_regression_r2(X[tr], y[tr], X[te], y[te])
            results[f"{prof_name}/{target_name}_r2"] = r2

    # Build summary plot — bar chart of probe scores
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = []
    scores = []
    bar_colors = []
    for k, v in results.items():
        labels.append(k)
        scores.append(v)
        bar_colors.append("#377eb8" if "r2" in k else "#4daf4a")
    bars = ax.bar(labels, scores, color=bar_colors, edgecolor="black", linewidth=0.7)
    for bar, sc in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            sc + 0.02,
            f"{sc:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    ax.set_ylabel("Probe score (R² for regression, accuracy for classification)")
    ax.set_title(
        "Linear probes on stem features: what does the stem encode about the channel?", fontsize=12, fontweight="bold"
    )
    ax.set_xticklabels(labels, rotation=30, ha="right")
    plt.tight_layout()
    out_path = out_dir / "router_mechanism_linear_probing.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[A] wrote {out_path}", file=sys.stderr)
    plt.close(fig)

    # Save raw scores
    (out_dir / "router_mechanism_linear_probing.json").write_text(json.dumps(results, indent=2))
    return results


def expert_specialization(data_uma: dict, data_tdlc: dict, out_dir: Path):
    """C: Distribution of channel features per chosen expert."""
    print("[C] Expert specialization analysis...", file=sys.stderr)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")
    for col, prof in enumerate(["uma", "tdlc"]):
        d = data_uma if prof == "uma" else data_tdlc
        # Top row: SNR distribution per expert
        for e in (0, 1, 2):
            mask = d["experts"] == e
            if mask.sum() == 0:
                continue
            axes[0, col].hist(
                d["snr_db"][mask],
                bins=20,
                alpha=0.6,
                label=EXPERT_NAMES[e],
                color=EXPERT_COLORS[EXPERT_NAMES[e]],
                edgecolor="black",
                linewidth=0.4,
            )
        axes[0, col].set_title(f"{prof.upper()}: SNR per chosen expert")
        axes[0, col].set_xlabel("SNR (dB)")
        if col == 0:
            axes[0, col].set_ylabel("Sample count")
        axes[0, col].legend(loc="upper right", fontsize=9)
        axes[0, col].grid(True, alpha=0.25)

        # Bottom row: BLER per expert per SNR bin
        snr_min, snr_max = d["snr_db"].min(), d["snr_db"].max()
        bins = np.linspace(snr_min, snr_max, 11)
        for e in (0, 1, 2):
            mask = d["experts"] == e
            if mask.sum() == 0:
                continue
            bler_per_bin = []
            bin_centers = []
            for i in range(len(bins) - 1):
                bm = mask & (d["snr_db"] >= bins[i]) & (d["snr_db"] < bins[i + 1])
                if bm.sum() < 5:
                    continue
                bler_per_bin.append(d["block_err"][bm].mean())
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
            axes[1, col].plot(
                bin_centers,
                bler_per_bin,
                "o-",
                color=EXPERT_COLORS[EXPERT_NAMES[e]],
                label=EXPERT_NAMES[e],
                linewidth=2,
                markersize=6,
            )
        axes[1, col].set_title(f"{prof.upper()}: BLER per expert across SNR bins")
        axes[1, col].set_xlabel("SNR (dB)")
        if col == 0:
            axes[1, col].set_ylabel("BLER")
        axes[1, col].set_ylim(0, 1.05)
        axes[1, col].legend(loc="upper right", fontsize=9)
        axes[1, col].grid(True, alpha=0.25)

    # Far right column: bar chart — what fraction of each expert's traffic is "wasted" (correctly predicted)?
    for col, prof in enumerate(["uma", "tdlc"]):
        # already did 2 cols above, that's the plot. We have 3 cols total.
        pass
    # Replace right column with summary
    for ax in axes[:, 2]:
        ax.set_visible(False)
    sum_ax = fig.add_subplot(1, 3, 3)
    bar_w = 0.35
    x = np.arange(3)
    uma_usage = np.array([(data_uma["experts"] == e).mean() for e in range(3)])
    tdlc_usage = np.array([(data_tdlc["experts"] == e).mean() for e in range(3)])
    sum_ax.bar(x - bar_w / 2, uma_usage * 100, bar_w, label="UMa", color="#5b9bd5", edgecolor="black", linewidth=0.6)
    sum_ax.bar(x + bar_w / 2, tdlc_usage * 100, bar_w, label="TDLC", color="#1f4e79", edgecolor="black", linewidth=0.6)
    for i, (u, t) in enumerate(zip(uma_usage, tdlc_usage)):
        sum_ax.text(i - bar_w / 2, u * 100 + 1.5, f"{u*100:.0f}%", ha="center", fontsize=9)
        sum_ax.text(i + bar_w / 2, t * 100 + 1.5, f"{t*100:.0f}%", ha="center", fontsize=9)
    sum_ax.set_xticks(x)
    sum_ax.set_xticklabels(EXPERT_NAMES)
    sum_ax.set_ylabel("Routing share (%)")
    sum_ax.set_title("Routing share per expert × profile")
    sum_ax.legend(loc="upper right", fontsize=9)
    sum_ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Per-expert specialization — what does each expert see, and what does it deliver?",
        fontsize=13,
        y=1.02,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = out_dir / "router_mechanism_expert_specialization.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[C] wrote {out_path}", file=sys.stderr)
    plt.close(fig)


def decision_boundary(data_uma: dict, data_tdlc: dict, out_dir: Path):
    """F: PCA → router decision regions overlaid with sample SNR coloring."""
    print("[F] Decision boundary visualization...", file=sys.stderr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (prof, d) in zip(axes, [("uma", data_uma), ("tdlc", data_tdlc)]):
        X = d["feats"]
        # PCA via SVD
        Xc = X - X.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
        embed = Xc @ vt[:2].T  # (N, 2)

        # Generate dense grid in PCA plane
        x_min, x_max = embed[:, 0].min(), embed[:, 0].max()
        y_min, y_max = embed[:, 1].min(), embed[:, 1].max()
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 80),
            np.linspace(y_min, y_max, 80),
        )
        grid_pca = np.stack([xx.ravel(), yy.ravel()], axis=1)
        # Lift back to feature space (using top-2 PCs)
        grid_feat = grid_pca @ vt[:2] + X.mean(axis=0)

        # For each grid point, find nearest neighbor's expert assignment
        # (cheap proxy for "what would the router do here" without re-running the NN)
        # Pure-numpy KNN: chunked broadcast distances to avoid OOM on the grid
        chunk = 500
        idx = np.empty((len(grid_feat), 5), dtype=np.int64)
        for start in range(0, len(grid_feat), chunk):
            end = min(start + chunk, len(grid_feat))
            d2 = ((grid_feat[start:end, None, :] - X[None, :, :]) ** 2).sum(axis=-1)
            idx[start:end] = np.argpartition(d2, 5, axis=-1)[:, :5]
        nn_experts = d["experts"][idx]
        # Vote: most common expert among 5 nearest neighbors
        grid_expert = np.array([np.bincount(row, minlength=3).argmax() for row in nn_experts]).reshape(xx.shape)

        # Color decision regions (faded)
        cmap = plt.matplotlib.colors.ListedColormap(
            [EXPERT_COLORS["nano"], EXPERT_COLORS["small"], EXPERT_COLORS["large"]]
        )
        ax.contourf(xx, yy, grid_expert, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap, alpha=0.35)

        # Overlay actual samples colored by SNR
        sc = ax.scatter(embed[:, 0], embed[:, 1], c=d["snr_db"], s=4, alpha=0.65, cmap="viridis", edgecolor="none")
        ax.set_title(f"{prof.upper()}: routing decision regions (background) + sample SNR (dots)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.colorbar(sc, ax=ax, label="True SNR (dB)")
        # Add expert legend
        from matplotlib.patches import Patch

        legend_elems = [Patch(facecolor=EXPERT_COLORS[n], alpha=0.5, label=n) for n in EXPERT_NAMES]
        ax.legend(handles=legend_elems, loc="upper right", title="Routed to", fontsize=9)

    fig.suptitle(
        "Router decision boundary in PCA space — does it align with SNR contours?",
        fontsize=13,
        y=1.02,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = out_dir / "router_mechanism_decision_boundary.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[F] wrote {out_path}", file=sys.stderr)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="knn_moe-5g-nrx/moe-5g-nrx/model-moe_alphasweep_asym_a2e3_s67-t6lkdep2:best",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/storage/brno2/home/ramsay/moe-5g-datasets/dense-v1/test"),
    )
    parser.add_argument("--max-samples", type=int, default=4000)
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

    print("[INFO] Collecting UMa data...", file=sys.stderr)
    loader_uma = build_cached_dataloader(path=args.data_dir / "uma.pt", batch_size=128, num_workers=0)
    data_uma = collect_data(model, loader_uma, device, args.max_samples)
    print("[INFO] Collecting TDLC data...", file=sys.stderr)
    loader_tdlc = build_cached_dataloader(path=args.data_dir / "tdlc.pt", batch_size=128, num_workers=0)
    data_tdlc = collect_data(model, loader_tdlc, device, args.max_samples)

    probe_results = linear_probing(data_uma, data_tdlc, args.out)
    print("\n=== Linear probing results ===")
    for k, v in probe_results.items():
        print(f"  {k}: {v:.4f}")

    expert_specialization(data_uma, data_tdlc, args.out)
    decision_boundary(data_uma, data_tdlc, args.out)
    print("\n[DONE] All three analyses complete. Figures in docs/figures/.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
