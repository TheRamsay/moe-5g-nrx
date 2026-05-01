"""Per-sample comparison: where does neural (exp26) beat classical (LMMSE)?

Two analyses:
  Analysis 1 — Per-sample crosstab. For each test sample, record both NN and
               LMMSE block-success. Build a 2x2 confusion matrix:
                  - both succeed
                  - NN only succeeds (← the "neural wins" subset we care about)
                  - LMMSE only succeeds
                  - both fail
               Characterize the "NN-only" subset by channel features
               (RMS delay, K-factor, spatial max eigenvalue, channel power, SNR).

  Analysis 2 — SNR slack. For each receiver, find the SNR threshold where
               BLER drops below {0.5, 0.1, 0.01}. Compare thresholds → spectral
               efficiency advantage.

Outputs:
  docs/figures/neural_vs_lmmse_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.lmmse import lmmse_forward  # noqa: E402
from src.data import build_cached_dataloader  # noqa: E402
from src.models import build_model_from_config  # noqa: E402


def _compute_features(channel_targets: torch.Tensor) -> dict[str, np.ndarray]:
    """Compute per-sample channel features for characterization.

    Args:
        channel_targets: (batch, 2*N_ant, freq, time) real/imag stacked channel.
    Returns:
        Dict of (batch,) numpy arrays.
    """
    n_ant = channel_targets.shape[1] // 2
    h_real = channel_targets[:, :n_ant]
    h_imag = channel_targets[:, n_ant:]
    h = torch.complex(h_real, h_imag)  # (B, n_ant, freq, time)

    # Channel power (mean |H|² across all antennas/freq/time)
    chan_power = (h_real**2 + h_imag**2).mean(dim=(1, 2, 3)).cpu().numpy()

    # RMS delay spread (FFT-based)
    cir = torch.fft.ifft(h, dim=2)
    pdp = (cir.real**2 + cir.imag**2).mean(dim=(1, 3))
    delays = torch.arange(pdp.shape[1], dtype=pdp.dtype, device=pdp.device)
    pdp_norm = pdp / pdp.sum(dim=1, keepdim=True).clamp_min(1e-12)
    tau_mean = (delays * pdp_norm).sum(dim=1)
    tau_rms = (((delays - tau_mean.unsqueeze(1)) ** 2) * pdp_norm).sum(dim=1).clamp_min(1e-12).sqrt()

    # K-factor (Rician LOS-to-NLOS ratio): power of mean h vs variance
    h_mean = h.mean(dim=(2, 3))  # (B, n_ant)
    h_dev = h - h_mean.unsqueeze(-1).unsqueeze(-1)
    pow_los = (h_mean.real**2 + h_mean.imag**2).mean(dim=1)
    pow_scatter = (h_dev.real**2 + h_dev.imag**2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    k_factor = (pow_los / pow_scatter).cpu().numpy()

    # Spatial correlation eigenvalue summary (max eigenvalue of antenna covariance)
    h_flat = h.reshape(h.shape[0], n_ant, -1)
    R = torch.einsum("bif,bjf->bij", h_flat, torch.conj(h_flat)) / h_flat.shape[-1]
    eigs = torch.linalg.eigvalsh(R).clamp_min(0)  # ascending
    spatial_max_eig = eigs[:, -1].cpu().numpy()

    return {
        "chan_power": chan_power,
        "rms_delay": tau_rms.cpu().numpy(),
        "k_factor": k_factor,
        "spatial_max_eig": spatial_max_eig,
    }


def collect_per_sample(model, loader, device, max_samples: int) -> dict:
    """Run BOTH NN and LMMSE on each sample, collect per-sample outcomes."""
    model.eval()
    rec: dict[str, list] = {
        "snr_db": [],
        "nn_bler": [],
        "nn_ber": [],
        "lmmse_bler": [],
        "lmmse_ber": [],
        "chan_power": [],
        "rms_delay": [],
        "k_factor": [],
        "spatial_max_eig": [],
    }

    n = 0
    with torch.inference_mode():
        for batch in loader:
            if n >= max_samples:
                break
            x = batch.inputs.to(device, non_blocking=True)
            targets = batch.bit_labels.to(device)
            channel_targets = batch.channel_target.to(device)
            snr_db = batch.snr_db.to(device)
            B = x.shape[0]
            n += B

            # --- NN forward (exp26 baseline routing) ---
            nn_out = model(x)
            nn_logits_flat = nn_out["logits"]  # (B, total_bits)
            nn_logits = nn_logits_flat.view(B, model.bits_per_symbol, model.num_subcarriers, model.num_ofdm_symbols)
            nn_preds = (nn_logits > 0).long()
            nn_wrong = nn_preds != targets.long()
            nn_ber = nn_wrong.float().mean(dim=(1, 2, 3)).cpu().numpy()
            nn_bler = nn_wrong.any(dim=(1, 2, 3)).long().cpu().numpy().astype(int)

            # --- LMMSE forward (LS-MRC) ---
            lmmse_out = lmmse_forward(x, snr_db, num_rx_antennas=4, mode="ls_mrc")
            lmmse_logits_flat = lmmse_out["logits"]
            lmmse_logits = lmmse_logits_flat.view(
                B, model.bits_per_symbol, model.num_subcarriers, model.num_ofdm_symbols
            )
            lmmse_preds = (lmmse_logits > 0).long()
            lmmse_wrong = lmmse_preds != targets.long()
            lmmse_ber = lmmse_wrong.float().mean(dim=(1, 2, 3)).cpu().numpy()
            lmmse_bler = lmmse_wrong.any(dim=(1, 2, 3)).long().cpu().numpy().astype(int)

            # --- Channel features ---
            feats = _compute_features(channel_targets)

            rec["snr_db"].append(snr_db.cpu().numpy())
            rec["nn_bler"].append(nn_bler)
            rec["nn_ber"].append(nn_ber)
            rec["lmmse_bler"].append(lmmse_bler)
            rec["lmmse_ber"].append(lmmse_ber)
            for k, v in feats.items():
                rec[k].append(v)

    return {k: np.concatenate(v, axis=0)[:max_samples] for k, v in rec.items() if v}


def analysis_1_crosstab(data: dict) -> dict:
    """Per-sample (NN_success, LMMSE_success) confusion matrix + feature characterization."""
    nn_success = (data["nn_bler"] == 0).astype(int)
    lmmse_success = (data["lmmse_bler"] == 0).astype(int)

    n_total = len(nn_success)
    both_succeed = ((nn_success == 1) & (lmmse_success == 1)).sum()
    nn_only = ((nn_success == 1) & (lmmse_success == 0)).sum()
    lmmse_only = ((nn_success == 0) & (lmmse_success == 1)).sum()
    both_fail = ((nn_success == 0) & (lmmse_success == 0)).sum()

    # Where NN is the better choice (matches when both succeed; wins where only NN succeeds)
    nn_strictly_wins_mask = (nn_success == 1) & (lmmse_success == 0)
    lmmse_strictly_wins_mask = (nn_success == 0) & (lmmse_success == 1)

    feature_keys = ["snr_db", "chan_power", "rms_delay", "k_factor", "spatial_max_eig"]

    def _stats(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        out = {"n": n}
        for k in feature_keys:
            v = data[k][mask]
            out[k] = {
                "mean": float(v.mean()),
                "p25": float(np.percentile(v, 25)),
                "p50": float(np.percentile(v, 50)),
                "p75": float(np.percentile(v, 75)),
            }
        return out

    return {
        "n_total": int(n_total),
        "confusion": {
            "both_succeed": int(both_succeed),
            "nn_only_wins": int(nn_only),
            "lmmse_only_wins": int(lmmse_only),
            "both_fail": int(both_fail),
        },
        "rates": {
            "nn_success_rate": float(nn_success.mean()),
            "lmmse_success_rate": float(lmmse_success.mean()),
            "agreement_rate": float((nn_success == lmmse_success).mean()),
        },
        "feature_stats_when_nn_wins": _stats(nn_strictly_wins_mask),
        "feature_stats_when_lmmse_wins": _stats(lmmse_strictly_wins_mask),
        "feature_stats_overall": _stats(np.ones(n_total, dtype=bool)),
    }


def analysis_2_snr_slack(data: dict, n_bins: int = 14, thresholds: list[float] | None = None) -> dict:
    """For each receiver, find SNR threshold where BLER < target."""
    if thresholds is None:
        thresholds = [0.5, 0.1, 0.05, 0.01]

    snr = data["snr_db"]
    edges = np.linspace(snr.min(), snr.max(), n_bins + 1)
    bin_idx = np.digitize(snr, edges[1:-1])
    centers = [(edges[b] + edges[b + 1]) / 2 for b in range(n_bins)]

    nn_bler_bins = []
    lmmse_bler_bins = []
    n_per_bin = []
    for b in range(n_bins):
        m = bin_idx == b
        nn_bler_bins.append(float(data["nn_bler"][m].mean()) if m.any() else float("nan"))
        lmmse_bler_bins.append(float(data["lmmse_bler"][m].mean()) if m.any() else float("nan"))
        n_per_bin.append(int(m.sum()))

    def _threshold_snr(bler_per_bin: list[float], threshold: float) -> float | None:
        """First SNR at which BLER drops below threshold; None if never."""
        for c, b in zip(centers, bler_per_bin):
            if not np.isnan(b) and b < threshold:
                return float(c)
        return None

    return {
        "snr_centers": centers,
        "n_per_bin": n_per_bin,
        "nn_bler_bins": nn_bler_bins,
        "lmmse_bler_bins": lmmse_bler_bins,
        "threshold_crossings": {
            f"target_{t}": {
                "nn_first_snr": _threshold_snr(nn_bler_bins, t),
                "lmmse_first_snr": _threshold_snr(lmmse_bler_bins, t),
            }
            for t in thresholds
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data-dir", type=Path, required=True, help="dir with uma.pt and tdlc.pt")
    ap.add_argument("--out-json", type=Path, default=PROJECT_ROOT / "docs" / "figures" / "neural_vs_lmmse_results.json")
    ap.add_argument("--max-samples", type=int, default=32768, help="per profile")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    print(f"[ckpt] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model_from_config(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    results = {}
    for profile in ["uma", "tdlc"]:
        print(f"\n=== {profile} ===")
        loader = build_cached_dataloader(
            path=args.data_dir / f"{profile}.pt",
            batch_size=128,
            num_workers=0,
        )
        data = collect_per_sample(model, loader, device, args.max_samples)
        a1 = analysis_1_crosstab(data)
        a2 = analysis_2_snr_slack(data)
        results[profile] = {"analysis_1_crosstab": a1, "analysis_2_snr_slack": a2}

        print(f"  N total: {a1['n_total']}")
        print(
            f"  Confusion (both/NN-only/LMMSE-only/both-fail): "
            f"{a1['confusion']['both_succeed']}/{a1['confusion']['nn_only_wins']}/"
            f"{a1['confusion']['lmmse_only_wins']}/{a1['confusion']['both_fail']}"
        )
        print(
            f"  NN succ {a1['rates']['nn_success_rate']:.4f}  "
            f"LMMSE succ {a1['rates']['lmmse_success_rate']:.4f}  "
            f"agreement {a1['rates']['agreement_rate']:.4f}"
        )
        if a1["feature_stats_when_nn_wins"]["n"] > 0:
            print(f"  When NN-only-wins (n={a1['feature_stats_when_nn_wins']['n']}):")
            for k in ["snr_db", "rms_delay", "k_factor", "spatial_max_eig"]:
                if k in a1["feature_stats_when_nn_wins"]:
                    s = a1["feature_stats_when_nn_wins"][k]
                    sov = a1["feature_stats_overall"][k]
                    print(f"    {k}: median {s['p50']:.3f} (overall median {sov['p50']:.3f})")
        if a1["feature_stats_when_lmmse_wins"]["n"] > 0:
            print(f"  When LMMSE-only-wins (n={a1['feature_stats_when_lmmse_wins']['n']}):")
            for k in ["snr_db", "rms_delay", "k_factor", "spatial_max_eig"]:
                if k in a1["feature_stats_when_lmmse_wins"]:
                    s = a1["feature_stats_when_lmmse_wins"][k]
                    sov = a1["feature_stats_overall"][k]
                    print(f"    {k}: median {s['p50']:.3f} (overall median {sov['p50']:.3f})")

        print("  Threshold-crossings (first SNR with BLER < target):")
        for tk, v in a2["threshold_crossings"].items():
            print(f"    {tk}: NN at {v['nn_first_snr']}, LMMSE at {v['lmmse_first_snr']}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[done] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
