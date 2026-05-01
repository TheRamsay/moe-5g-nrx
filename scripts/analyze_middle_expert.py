"""Middle-expert investigation: what does small ACTUALLY contribute?

Loads exp26 checkpoint and on the test set:
  - Captures stem features per sample
  - Records the router's chosen expert
  - Forces each expert (counterfactually) on every sample → records logits + channel estimate
  - Computes per-sample, per-expert: BER, BLER, channel MSE, mean bit confidence

Answers the five investigation questions:
  Q1: BER on small's routed samples (is it random or partial decode?)
  Q2: Counterfactual — what would large/nano/sink have done on small's samples?
  Q3: Channel MSE per expert per routing — does small produce useful estimates?
  Q4: Per-SNR per-expert success rate (is "0%" universal or just average?)
  Q5: Bit-confidence histogram per expert (random vs confident-wrong vs near-correct)

Outputs:
  docs/figures/middle_expert_q1_ber_per_route.png
  docs/figures/middle_expert_q2_counterfactual.png
  docs/figures/middle_expert_q3_channel_mse.png
  docs/figures/middle_expert_q4_per_snr_success.png
  docs/figures/middle_expert_q5_confidence_hist.png
  docs/figures/middle_expert_results.json
"""

from __future__ import annotations

import argparse
import json
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

EXPERT_NAMES = ["nano", "small", "large"]
EXPERT_COLORS = {"nano": "#fdc086", "small": "#7fc97f", "large": "#beaed4"}
PROFILE_COLORS = {"uma": "#1f77b4", "tdlc": "#d62728"}


def _download_artifact(ref: str) -> Path:
    import wandb

    api = wandb.Api()
    art = api.artifact(ref, type="model")
    art_dir = Path(art.download(root=f"/tmp/wandb_middle/{ref.replace('/', '_').replace(':', '_')}"))
    for path in art_dir.rglob("*.pt"):
        return path
    raise FileNotFoundError(f"No .pt file in {art_dir}")


def collect_per_expert_outputs(model, loader, device, max_samples: int) -> dict:
    """Forward pass with COUNTERFACTUAL expert evaluation.

    For each batch, after the stem, we call ALL experts on the same shared_state.
    Records per-sample outputs from each expert independently of router choice.
    """
    model.eval()
    captured = {}

    def hook(_module, _input, output):
        captured["shared_state"] = output

    handle = model.stem.register_forward_hook(hook)

    n_collected = 0
    rec = {
        "router_choice": [],  # (N,) — which expert the router picked (idx into expert_names)
        "snr_db": [],  # (N,)
        # Per-expert per-sample metrics
        "ber": {n: [] for n in EXPERT_NAMES},  # (N,) — fraction of bits wrong
        "block_err": {n: [] for n in EXPERT_NAMES},  # (N,) — 1 if any bit wrong else 0
        "chan_mse": {n: [] for n in EXPERT_NAMES},  # (N,) — channel MSE
        "mean_conf": {n: [] for n in EXPERT_NAMES},  # (N,) — mean |sigmoid(logits) - 0.5|
        "wrong_conf": {n: [] for n in EXPERT_NAMES},  # (N,) — mean |sigmoid(logits) - 0.5| on WRONG bits only
    }

    with torch.inference_mode():
        for batch in loader:
            if n_collected >= max_samples:
                break
            x = batch.inputs.to(device, non_blocking=True)
            targets = batch.bit_labels.to(device)  # (B, bits, freq, time)
            channel_targets = batch.channel_target.to(device)  # (B, 2*N_ant, freq, time)
            snr_db = batch.snr_db.cpu().numpy()

            # Run model so router decision happens; we hook the stem output
            out = model(x)
            shared_state = captured["shared_state"]
            router_choice = out["selected_expert_index"].cpu().numpy()

            B = x.shape[0]
            n_collected += B

            rec["router_choice"].append(router_choice)
            rec["snr_db"].append(snr_db)

            # Counterfactual: call each expert directly on the SAME shared_state
            for name in EXPERT_NAMES:
                expert_logits, expert_channel = model.experts[name](shared_state)
                # expert_logits: (B, bits_per_symbol, freq, time)
                # targets: (B, bits, freq, time) — same shape

                # Predictions
                preds = (expert_logits > 0).long()
                wrong = preds != targets.long()  # (B, bits, freq, time)
                # BER = fraction of wrong bits per sample
                ber = wrong.float().mean(dim=(1, 2, 3)).cpu().numpy()
                rec["ber"][name].append(ber)
                # Block error = any wrong bit per sample
                block_err = wrong.any(dim=(1, 2, 3)).long().cpu().numpy().astype(int)
                rec["block_err"][name].append(block_err)
                # Channel MSE per sample
                cmse = ((expert_channel - channel_targets) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
                rec["chan_mse"][name].append(cmse)
                # Mean bit confidence: |sigmoid(logits) - 0.5|, range [0, 0.5]; near 0 = uncertain
                probs = torch.sigmoid(expert_logits)
                conf = (probs - 0.5).abs().mean(dim=(1, 2, 3)).cpu().numpy()
                rec["mean_conf"][name].append(conf)
                # Confidence on WRONG bits — high = "confidently wrong"
                wrong_mask = wrong.float()
                wrong_count = wrong_mask.sum(dim=(1, 2, 3)).clamp_min(1)
                wrong_conf = ((probs - 0.5).abs() * wrong_mask).sum(dim=(1, 2, 3)) / wrong_count
                rec["wrong_conf"][name].append(wrong_conf.cpu().numpy())

    handle.remove()

    out_dict = {
        "router_choice": np.concatenate(rec["router_choice"], axis=0)[:max_samples],
        "snr_db": np.concatenate(rec["snr_db"], axis=0)[:max_samples],
    }
    for k in ["ber", "block_err", "chan_mse", "mean_conf", "wrong_conf"]:
        out_dict[k] = {n: np.concatenate(rec[k][n], axis=0)[:max_samples] for n in EXPERT_NAMES}
    return out_dict


def aggregate_q1(data: dict) -> dict:
    """Q1: Per-expert BER on the samples THE ROUTER ROUTED to that expert."""
    out = {}
    for ei, name in enumerate(EXPERT_NAMES):
        mask = data["router_choice"] == ei
        n = int(mask.sum())
        if n == 0:
            out[name] = {
                "n": 0,
                "ber_mean": float("nan"),
                "bler": float("nan"),
                "ber_p25": float("nan"),
                "ber_p50": float("nan"),
                "ber_p75": float("nan"),
            }
            continue
        ber = data["ber"][name][mask]
        bler = data["block_err"][name][mask]
        out[name] = {
            "n": n,
            "ber_mean": float(ber.mean()),
            "bler": float(bler.mean()),
            "ber_p25": float(np.percentile(ber, 25)),
            "ber_p50": float(np.percentile(ber, 50)),
            "ber_p75": float(np.percentile(ber, 75)),
        }
    return out


def aggregate_q2(data: dict) -> dict:
    """Q2: Counterfactual — for samples routed to expert X, what BER/BLER would Y have achieved?"""
    out = {}
    for ei, route_name in enumerate(EXPERT_NAMES):
        mask = data["router_choice"] == ei
        n = int(mask.sum())
        if n == 0:
            out[route_name] = {"n": 0}
            continue
        per_alt = {}
        for alt_name in EXPERT_NAMES:
            ber_alt = data["ber"][alt_name][mask]
            bler_alt = data["block_err"][alt_name][mask]
            per_alt[alt_name] = {
                "ber_mean": float(ber_alt.mean()),
                "bler": float(bler_alt.mean()),
            }
        out[route_name] = {"n": n, "alternatives": per_alt}
    return out


def aggregate_q3(data: dict) -> dict:
    """Q3: Channel MSE per expert on routed samples (and counterfactually)."""
    out = {}
    for ei, route_name in enumerate(EXPERT_NAMES):
        mask = data["router_choice"] == ei
        n = int(mask.sum())
        if n == 0:
            out[route_name] = {"n": 0}
            continue
        per_alt = {}
        for alt_name in EXPERT_NAMES:
            mse = data["chan_mse"][alt_name][mask]
            per_alt[alt_name] = {
                "mse_mean": float(mse.mean()),
                "mse_p50": float(np.percentile(mse, 50)),
            }
        out[route_name] = {"n": n, "channel_mse_by_expert": per_alt}
    return out


def aggregate_q4(data: dict, n_bins: int = 7) -> dict:
    """Q4: Per-SNR success rate per expert (forced routing — what each can decode at each SNR)."""
    snr = data["snr_db"]
    bins = np.linspace(snr.min(), snr.max(), n_bins + 1)
    bin_idx = np.digitize(snr, bins[1:-1])
    out = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        lo = float(bins[b])
        hi = float(bins[b + 1])
        per_expert = {}
        for name in EXPERT_NAMES:
            if n == 0:
                per_expert[name] = {"success_rate": float("nan"), "ber": float("nan")}
            else:
                bler = data["block_err"][name][mask]
                per_expert[name] = {
                    "success_rate": float(1.0 - bler.mean()),
                    "ber": float(data["ber"][name][mask].mean()),
                }
        out.append({"snr_lo": lo, "snr_hi": hi, "n": n, "experts": per_expert})
    return out


def aggregate_q5(data: dict) -> dict:
    """Q5: Confidence stats per expert on routed samples."""
    out = {}
    for ei, route_name in enumerate(EXPERT_NAMES):
        mask = data["router_choice"] == ei
        n = int(mask.sum())
        if n == 0:
            out[route_name] = {"n": 0}
            continue
        out[route_name] = {
            "n": n,
            "mean_confidence": float(data["mean_conf"][route_name][mask].mean()),
            "wrong_bit_confidence": float(data["wrong_conf"][route_name][mask].mean()),
        }
    return out


def plot_q1(profile_results: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, profile in zip(axes, ["uma", "tdlc"]):
        d = profile_results[profile]["q1"]
        names = [n for n in EXPERT_NAMES if d[n]["n"] > 0]
        bers = [d[n]["ber_mean"] for n in names]
        ax.bar(names, bers, color=[EXPERT_COLORS[n] for n in names])
        for i, n in enumerate(names):
            ax.text(
                i, bers[i] + 0.01, f"BER={bers[i]:.3f}\nBLER={d[n]['bler']:.3f}\nN={d[n]['n']}", ha="center", fontsize=9
            )
        ax.set_title(f"{profile.upper()} — BER/BLER on router-routed samples")
        ax.set_ylabel("BER (mean over routed samples)")
        ax.set_ylim(0, max(0.55, max(bers) * 1.15) if bers else 0.55)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="random (0.5)")
        ax.legend()
    fig.suptitle("Q1: What's the actual BER each expert produces on its routed samples?", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_q2(profile_results: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, profile in zip(axes, ["uma", "tdlc"]):
        d = profile_results[profile]["q2"]
        x = np.arange(len(EXPERT_NAMES))
        width = 0.27
        for i, alt in enumerate(EXPERT_NAMES):
            vals = [d[route]["alternatives"][alt]["bler"] if d[route]["n"] > 0 else 0 for route in EXPERT_NAMES]
            ax.bar(x + (i - 1) * width, vals, width, label=f"{alt} forced", color=EXPERT_COLORS[alt])
        ax.set_xticks(x)
        ax.set_xticklabels([f"router→{r}\n(N={d[r]['n']})" for r in EXPERT_NAMES])
        ax.set_ylabel("BLER (block error rate)")
        ax.set_title(f"{profile.upper()} — counterfactual: what would each expert have done?")
        ax.legend(loc="lower left", fontsize=8)
        ax.set_ylim(0, 1.05)
    fig.suptitle("Q2: For samples the router sent to X, what would Y have achieved?", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_q3(profile_results: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, profile in zip(axes, ["uma", "tdlc"]):
        d = profile_results[profile]["q3"]
        x = np.arange(len(EXPERT_NAMES))
        width = 0.27
        for i, alt in enumerate(EXPERT_NAMES):
            vals = [
                d[route]["channel_mse_by_expert"][alt]["mse_mean"] if d[route]["n"] > 0 else 0 for route in EXPERT_NAMES
            ]
            ax.bar(x + (i - 1) * width, vals, width, label=f"{alt} forced", color=EXPERT_COLORS[alt])
        ax.set_xticks(x)
        ax.set_xticklabels([f"router→{r}\n(N={d[r]['n']})" for r in EXPERT_NAMES])
        ax.set_ylabel("Channel MSE (lower = better)")
        ax.set_yscale("log")
        ax.set_title(f"{profile.upper()} — channel-estimate quality per expert")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Q3: Does small produce useful channel estimates that nano/sink couldn't?", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_q4(profile_results: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, profile in zip(axes, ["uma", "tdlc"]):
        bins_data = profile_results[profile]["q4"]
        snr_centers = [(b["snr_lo"] + b["snr_hi"]) / 2 for b in bins_data]
        for name in EXPERT_NAMES:
            sr = [b["experts"][name]["success_rate"] for b in bins_data]
            ax.plot(snr_centers, sr, marker="o", label=name, color=EXPERT_COLORS[name], linewidth=2)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Block success rate (1 - BLER)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{profile.upper()} — per-SNR success when FORCED to each expert")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Q4: Is small's 0% success universal, or only at low SNR?", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_q5(profile_results: dict, all_data: dict, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=False)
    for col, profile in enumerate(["uma", "tdlc"]):
        data = all_data[profile]
        for row, metric_key in enumerate(["mean_conf", "wrong_conf"]):
            ax = axes[row, col]
            for ei, name in enumerate(EXPERT_NAMES):
                mask = data["router_choice"] == ei
                if mask.sum() == 0:
                    continue
                vals = data[metric_key][name][mask]
                ax.hist(vals, bins=40, alpha=0.5, label=f"router→{name}", color=EXPERT_COLORS[name])
            ax.set_xlabel("Mean |sigmoid(logits) - 0.5|  (0=random, 0.5=fully confident)")
            label = "All bits" if metric_key == "mean_conf" else "Wrong bits only"
            ax.set_ylabel("Count")
            ax.set_title(f"{profile.upper()} — bit confidence ({label})")
            ax.legend()
    fig.suptitle("Q5: Are predictions random (0), correct-and-confident (high), or confident-wrong?", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True, help="Local path or W&B artifact ref to exp26 best.pt")
    ap.add_argument("--data-dir", type=Path, required=True, help="Test set dir with uma.pt and tdlc.pt")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "docs" / "figures")
    ap.add_argument("--max-samples", type=int, default=4000, help="Per profile (uma/tdlc).")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ckpt] not local — downloading W&B artifact: {args.checkpoint}")
        ckpt_path = _download_artifact(args.checkpoint)
    print(f"[ckpt] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model_from_config(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    profile_results: dict = {}
    all_data: dict = {}
    for profile in ["uma", "tdlc"]:
        print(f"\n=== {profile} ===")
        loader = build_cached_dataloader(
            path=args.data_dir / f"{profile}.pt",
            batch_size=128,
            num_workers=0,
        )
        data = collect_per_expert_outputs(model, loader, device, args.max_samples)
        all_data[profile] = data
        profile_results[profile] = {
            "q1": aggregate_q1(data),
            "q2": aggregate_q2(data),
            "q3": aggregate_q3(data),
            "q4": aggregate_q4(data),
            "q5": aggregate_q5(data),
        }
        print(f"  collected {len(data['snr_db'])} samples")
        print("  Q1 (BER on routed):")
        for name in EXPERT_NAMES:
            d = profile_results[profile]["q1"][name]
            if d["n"] > 0:
                print(f"    {name}: n={d['n']:5d}  BER={d['ber_mean']:.4f}  BLER={d['bler']:.4f}")
        print("  Q2 (counterfactual BLER for router→small samples):")
        d2 = profile_results[profile]["q2"].get("small", {})
        if d2.get("n", 0) > 0:
            for alt in EXPERT_NAMES:
                v = d2["alternatives"][alt]
                print(f"    if forced to {alt}: BER={v['ber_mean']:.4f}  BLER={v['bler']:.4f}")

    # Save plots + JSON
    plot_q1(profile_results, args.out_dir / "middle_expert_q1_ber_per_route.png")
    plot_q2(profile_results, args.out_dir / "middle_expert_q2_counterfactual.png")
    plot_q3(profile_results, args.out_dir / "middle_expert_q3_channel_mse.png")
    plot_q4(profile_results, args.out_dir / "middle_expert_q4_per_snr_success.png")
    plot_q5(profile_results, all_data, args.out_dir / "middle_expert_q5_confidence_hist.png")

    json_path = args.out_dir / "middle_expert_results.json"
    with open(json_path, "w") as f:
        json.dump(profile_results, f, indent=2, default=float)
    print(f"\n[done] wrote 5 figures + {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
