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


def _compute_advanced_channel_features(channel_targets: torch.Tensor) -> dict[str, np.ndarray]:
    """Compute physical channel features from the ground-truth H tensor.

    Args:
        channel_targets: (batch, 2*N, freq, time) real/imag stacked channel.
    Returns:
        Dict of (batch,) numpy arrays for each feature.
    """
    n_ant = channel_targets.shape[1] // 2
    h_real = channel_targets[:, :n_ant]
    h_imag = channel_targets[:, n_ant:]
    h = torch.complex(h_real, h_imag)  # (batch, n_ant, freq, time)

    # A. Real RMS delay spread via IFFT along freq → CIR → power-delay-profile moments
    cir = torch.fft.ifft(h, dim=2)  # (batch, n_ant, freq, time)
    pdp = (cir.real**2 + cir.imag**2).mean(dim=(1, 3))  # (batch, freq) power-delay-profile
    delays = torch.arange(pdp.shape[1], dtype=pdp.dtype, device=pdp.device)
    pdp_norm = pdp / pdp.sum(dim=1, keepdim=True).clamp_min(1e-12)
    tau_mean = (delays * pdp_norm).sum(dim=1)
    tau_rms = (((delays - tau_mean.unsqueeze(1)) ** 2) * pdp_norm).sum(dim=1).clamp_min(1e-12).sqrt()

    # B. Coherence bandwidth ≈ 1 / (5 · τ_rms)  (in normalised delay-bin units)
    coh_bw = 1.0 / (5.0 * tau_rms.clamp_min(1e-6))

    # C. Spatial correlation across antennas — eigenvalue summary of R = h h^H / N
    h_flat = h.reshape(h.shape[0], n_ant, -1)  # (batch, n_ant, freq*time)
    R = torch.einsum("bif,bjf->bij", h_flat, torch.conj(h_flat)) / h_flat.shape[-1]
    # R is Hermitian by construction; eigvalsh returns real ascending eigenvalues
    eigs = torch.linalg.eigvalsh(R).clamp_min(0)  # (batch, n_ant)
    spatial_trace = eigs.sum(dim=1)
    spatial_max_eig = eigs[:, -1]
    spatial_min_eig = eigs[:, 0]
    spatial_cond = spatial_max_eig / spatial_min_eig.clamp_min(1e-12)
    spatial_det = eigs.prod(dim=1)

    # D. K-factor (Rician LOS power ratio): peak CIR tap power / mean tap power
    pdp_per_sample = (cir.real**2 + cir.imag**2).mean(dim=1).mean(dim=2)  # (batch, freq) over ant+time
    peak_pow = pdp_per_sample.max(dim=1).values
    mean_pow = pdp_per_sample.mean(dim=1).clamp_min(1e-12)
    k_factor = peak_pow / mean_pow

    # E. Doppler / time-variation rate: variance of channel magnitude across time
    h_pow = (h.real**2 + h.imag**2).sum(dim=1)  # (batch, freq, time)
    doppler_proxy = h_pow.var(dim=2).mean(dim=1)  # variance across time, mean over freq

    return {
        "rms_delay_spread": tau_rms.cpu().numpy(),
        "coherence_bw": coh_bw.cpu().numpy(),
        "spatial_trace": spatial_trace.cpu().numpy(),
        "spatial_max_eig": spatial_max_eig.cpu().numpy(),
        "spatial_cond": spatial_cond.cpu().numpy(),
        "spatial_det": spatial_det.cpu().numpy(),
        "k_factor": k_factor.cpu().numpy(),
        "doppler_proxy": doppler_proxy.cpu().numpy(),
    }


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
    router_probs_list: list[np.ndarray] = []
    router_entropy_list: list[np.ndarray] = []
    advanced_features: dict[str, list[np.ndarray]] = {}
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
            h_mag = (h_real**2 + h_imag**2).sum(dim=1).cpu().numpy()  # (B, freq, time)
            ds = h_mag.var(axis=1).mean(axis=1)
            delay_spreads.append(ds)

            # Advanced channel features (RMS delay, coherence BW, spatial corr, K-factor, Doppler)
            adv = _compute_advanced_channel_features(channel_targets)
            for k, v in adv.items():
                advanced_features.setdefault(k, []).append(v)

            # Router probs + entropy (F + G)
            rp = out.get("router_probs")
            if rp is not None:
                rp_np = rp.cpu().numpy()
                router_probs_list.append(rp_np)
                # entropy per sample
                rp_clamp = np.clip(rp_np, 1e-12, 1.0)
                ent = -(rp_clamp * np.log(rp_clamp)).sum(axis=1)
                router_entropy_list.append(ent)

            # Per-sample block errors (any bit wrong)
            preds = (out["logits"] > 0).long()
            sample_block_err = (preds != targets.long()).any(dim=1).cpu().numpy().astype(int)
            block_errors.append(sample_block_err)

            n_collected += x.shape[0]

    handle.remove()
    result = {
        "feats": np.concatenate(feats, axis=0)[:max_samples],
        "experts": np.concatenate(experts, axis=0)[:max_samples],
        "snr_db": np.concatenate(snrs, axis=0)[:max_samples],
        "chan_power": np.concatenate(chan_powers, axis=0)[:max_samples],
        "delay_spread": np.concatenate(delay_spreads, axis=0)[:max_samples],
        "block_err": np.concatenate(block_errors, axis=0)[:max_samples],
    }
    for k, vs in advanced_features.items():
        result[k] = np.concatenate(vs, axis=0)[:max_samples]
    if router_probs_list:
        result["router_probs"] = np.concatenate(router_probs_list, axis=0)[:max_samples]
        result["router_entropy"] = np.concatenate(router_entropy_list, axis=0)[:max_samples]
    return result


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

    # Per-profile probes for physical channel features (regression)
    physical_targets = [
        # Original three
        ("snr", "snr_db"),
        ("chan_power", "chan_power"),
        ("delay_spread", "delay_spread"),
        # New physical features (A-E)
        ("rms_delay", "rms_delay_spread"),
        ("coherence_bw", "coherence_bw"),
        ("spatial_trace", "spatial_trace"),
        ("spatial_max_eig", "spatial_max_eig"),
        ("spatial_cond", "spatial_cond"),
        ("k_factor", "k_factor"),
        ("doppler", "doppler_proxy"),
    ]
    for prof_name, d in [("uma", data_uma), ("tdlc", data_tdlc)]:
        X = d["feats"]
        n = len(X)
        perm = rng.permutation(n)
        split = int(0.7 * n)
        tr, te = perm[:split], perm[split:]
        for target_name, key in physical_targets:
            if key not in d:
                continue
            y = d[key]
            r2 = _linear_regression_r2(X[tr], y[tr], X[te], y[te])
            results[f"{prof_name}/{target_name}_r2"] = r2

        # F. Probe router probabilities (one per expert)
        if "router_probs" in d:
            rp = d["router_probs"]
            for e_idx, e_name in enumerate(EXPERT_NAMES):
                r2 = _linear_regression_r2(X[tr], rp[tr, e_idx], X[te], rp[te, e_idx])
                results[f"{prof_name}/p_{e_name}_r2"] = r2

        # G. Probe router entropy
        if "router_entropy" in d:
            r2 = _linear_regression_r2(X[tr], d["router_entropy"][tr], X[te], d["router_entropy"][te])
            results[f"{prof_name}/router_entropy_r2"] = r2

    # Build summary plot — group bars by probe category, side-by-side UMa vs TDLC
    # Group results by target name (strip "uma/" or "tdlc/" prefix)
    grouped: dict[str, dict[str, float]] = {}
    for k, v in results.items():
        if "/" in k:
            prof, name = k.split("/", 1)
            grouped.setdefault(name, {})[prof] = v
        else:
            grouped.setdefault(k, {})["combined"] = v

    fig, ax = plt.subplots(figsize=(15, 6))
    target_names = list(grouped.keys())
    x = np.arange(len(target_names))
    bw = 0.4
    uma_vals = [grouped[t].get("uma", grouped[t].get("combined", 0.0)) for t in target_names]
    tdlc_vals = [grouped[t].get("tdlc", grouped[t].get("combined", 0.0)) for t in target_names]
    bars_u = ax.bar(x - bw / 2, uma_vals, bw, label="UMa", color="#5b9bd5", edgecolor="black", linewidth=0.6)
    bars_t = ax.bar(x + bw / 2, tdlc_vals, bw, label="TDLC", color="#1f4e79", edgecolor="black", linewidth=0.6)
    for bar, sc in zip(bars_u, uma_vals):
        if sc > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, sc + 0.015, f"{sc:.2f}", ha="center", fontsize=8)
    for bar, sc in zip(bars_t, tdlc_vals):
        if sc > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, sc + 0.015, f"{sc:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    ax.set_ylabel("Probe R² (regression) / accuracy (profile_acc)")
    ax.set_title("Linear probes on stem features — UMa vs TDLC, all targets", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(target_names, rotation=35, ha="right", fontsize=9)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
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


def expert_success_rate(data_uma: dict, data_tdlc: dict, out_dir: Path):
    """Per-expert success rate (block_err == 0) per fine SNR bin + summary table.

    Answers: "how often does each expert actually succeed on the samples it gets?"
    Resolves the apparent paradox of small showing BLER=1.0 across all bins yet
    being non-redundant per the exp41 ablation.
    """
    print("[D] Per-expert success-rate analysis...", file=sys.stderr)

    summary_rows: list[dict] = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (prof, d) in zip(axes, [("uma", data_uma), ("tdlc", data_tdlc)]):
        snr_min, snr_max = d["snr_db"].min(), d["snr_db"].max()
        # finer bins than the standard 7 — 14 bins gives ~2 dB resolution
        bins = np.linspace(snr_min, snr_max, 15)
        for e in (0, 1, 2):
            mask = d["experts"] == e
            if mask.sum() == 0:
                continue
            success_rate = []
            bin_centers = []
            n_per_bin = []
            for i in range(len(bins) - 1):
                bm = mask & (d["snr_db"] >= bins[i]) & (d["snr_db"] < bins[i + 1])
                n = int(bm.sum())
                if n < 3:
                    continue
                # success_rate = fraction of block_err == 0 (i.e. all bits correct)
                sr = 1.0 - d["block_err"][bm].mean()
                success_rate.append(sr)
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                n_per_bin.append(n)
            ax.plot(
                bin_centers,
                success_rate,
                "o-",
                color=EXPERT_COLORS[EXPERT_NAMES[e]],
                label=f"{EXPERT_NAMES[e]} (n={mask.sum()})",
                linewidth=2,
                markersize=6,
            )
            # Aggregate success-rate (mean across all samples this expert got)
            agg_sr = float(1.0 - d["block_err"][mask].mean())
            summary_rows.append(
                {
                    "profile": prof,
                    "expert": EXPERT_NAMES[e],
                    "n_samples": int(mask.sum()),
                    "agg_success_rate_pct": round(100 * agg_sr, 3),
                    "agg_bler": round(1.0 - agg_sr, 4),
                }
            )
        ax.set_title(f"{prof.upper()}: per-expert SUCCESS RATE per SNR bin (zoomed Y)")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Success rate (fraction of fully-correct blocks)")
        # zoom Y axis aggressively so even 1-5% successes are visible
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Per-expert SUCCESS RATE — does small ever decode? (linear log scale to surface low rates)",
        fontsize=13,
        y=1.02,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = out_dir / "router_mechanism_success_rate.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[D] wrote {out_path}", file=sys.stderr)
    plt.close(fig)

    # Save aggregate table as JSON for easy reference
    summary_path = out_dir / "router_mechanism_success_rate.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2))
    print("[D] aggregate per-expert success rates:", file=sys.stderr)
    for row in summary_rows:
        print(
            f"  {row['profile']}/{row['expert']}: n={row['n_samples']:5d}, "
            f"success_rate={row['agg_success_rate_pct']:.2f}%",
            file=sys.stderr,
        )


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
    expert_success_rate(data_uma, data_tdlc, args.out)
    print("\n[DONE] All four analyses complete. Figures in docs/figures/.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
