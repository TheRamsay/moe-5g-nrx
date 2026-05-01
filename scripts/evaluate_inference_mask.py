"""Inference-time routing modifications on a trained MoE checkpoint.

Two experiments, no retraining:
  Mode A: MASK small at inference. Router forced to argmax over {nano, large}.
          Tests "train with 3 experts, infer with 2 — discard small as scaffold."
  Mode B: REPLACE small with sink at inference. If router picks small, output
          zeros and charge zero FLOPs. Same router decisions, but small's slot
          becomes free.

For each mode, computes per-sample BER, BLER, realized FLOPs, and full routing
distribution per profile (uma + tdlc).

Outputs:
  docs/figures/inference_mask_results.json
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

from src.data import build_cached_dataloader  # noqa: E402
from src.models import build_model_from_config  # noqa: E402

EXPERT_NAMES = ["nano", "small", "large"]
SMALL_IDX = 1  # in [nano, small, large]


def _download_artifact(ref: str) -> Path:
    import wandb

    api = wandb.Api()
    art = api.artifact(ref, type="model")
    art_dir = Path(art.download(root=f"/tmp/wandb_mask/{ref.replace('/', '_').replace(':', '_')}"))
    for path in art_dir.rglob("*.pt"):
        return path
    raise FileNotFoundError(f"No .pt file in {art_dir}")


def evaluate_with_mode(model, loader, device, mode: str, max_samples: int) -> dict:
    """Run inference with routing modification.

    mode='baseline' — exp26 unchanged (sanity check)
    mode='A_mask'   — mask small's router logit, force argmax over {nano, large}
    mode='B_sink'   — keep router's choice but replace small's expert output with zeros
                      and charge zero FLOPs for small-routed samples
    """
    assert mode in ("baseline", "A_mask", "B_sink")
    model.eval()

    captured = {}

    def hook(_module, _input, output):
        captured["shared_state"] = output

    handle = model.stem.register_forward_hook(hook)

    n = 0
    bers = []
    blers = []
    flops_per_sample = []
    snrs = []
    routing_counts = {name: 0 for name in EXPERT_NAMES}
    routing_counts["sink_redirect"] = 0  # mode B: how many samples got the zero treatment

    base_flops = float(model.base_flops.item())
    expert_flops_full = model.expert_flops.cpu().numpy()  # [nano_only, small_only, large_only] AFTER stem

    with torch.inference_mode():
        for batch in loader:
            if n >= max_samples:
                break
            x = batch.inputs.to(device, non_blocking=True)
            targets = batch.bit_labels.to(device)
            B = x.shape[0]
            n += B

            # Forward through model — captures stem and router via the model's normal path
            out = model(x)
            shared_state = captured["shared_state"]
            router_logits = out["router_logits"]  # (B, 3)

            # Modify routing per mode
            if mode == "baseline":
                selected_expert_index = router_logits.argmax(dim=-1)
            elif mode == "A_mask":
                # Mask small to -inf so argmax picks from {nano, large}
                masked_logits = router_logits.clone()
                masked_logits[:, SMALL_IDX] = float("-inf")
                selected_expert_index = masked_logits.argmax(dim=-1)
            elif mode == "B_sink":
                # Keep router's natural argmax; small-routed samples get zero output
                selected_expert_index = router_logits.argmax(dim=-1)

            # Re-run forward via experts using the (possibly modified) selection
            # We mimic _run_top1_inference: per-expert mask -> apply expert
            bits_per_symbol = model.bits_per_symbol
            num_subcarriers = model.num_subcarriers
            num_ofdm_symbols = model.num_ofdm_symbols

            logits = torch.zeros(
                B, bits_per_symbol, num_subcarriers, num_ofdm_symbols, device=device, dtype=shared_state.dtype
            )

            # Track per-sample FLOPs charge
            sample_flops = np.full(B, base_flops, dtype=np.float64)

            for ei, name in enumerate(EXPERT_NAMES):
                mask = selected_expert_index == ei
                if not mask.any():
                    continue
                count = int(mask.sum().item())
                routing_counts[name] += count
                if mode == "B_sink" and ei == SMALL_IDX:
                    # Replace small with zero output AND zero FLOPs for these samples
                    routing_counts["sink_redirect"] += count
                    # logits stay zero (already initialized)
                    # FLOPs stay at base_flops only (no expert work)
                    continue
                # Run actual expert
                expert_logits, _ = model.experts[name](shared_state[mask])
                logits[mask] = expert_logits
                sample_flops[mask.cpu().numpy()] += float(expert_flops_full[ei])

            # Compute per-sample BER, BLER
            preds = (logits > 0).long()
            wrong = (preds != targets.long()).float()
            ber = wrong.mean(dim=(1, 2, 3)).cpu().numpy()
            bler = (wrong.sum(dim=(1, 2, 3)) > 0).long().cpu().numpy().astype(int)

            bers.append(ber)
            blers.append(bler)
            flops_per_sample.append(sample_flops)
            snrs.append(batch.snr_db.cpu().numpy())

    handle.remove()

    bers = np.concatenate(bers, axis=0)[:max_samples]
    blers = np.concatenate(blers, axis=0)[:max_samples]
    flops_per_sample = np.concatenate(flops_per_sample, axis=0)[:max_samples]
    snrs = np.concatenate(snrs, axis=0)[:max_samples]

    max_flops = float(model.max_flops.item())
    n_actual = len(bers)

    # Per-SNR binning (7 bins across the SNR range, like the dense-eval pipeline)
    n_bins = 7
    edges = np.linspace(snrs.min(), snrs.max(), n_bins + 1)
    bin_idx = np.digitize(snrs, edges[1:-1])
    per_snr = []
    for b in range(n_bins):
        m = bin_idx == b
        nb = int(m.sum())
        if nb == 0:
            per_snr.append(
                {
                    "snr_lo": float(edges[b]),
                    "snr_hi": float(edges[b + 1]),
                    "n": 0,
                    "bler": float("nan"),
                    "ber": float("nan"),
                    "realized_flops_ratio": float("nan"),
                }
            )
        else:
            per_snr.append(
                {
                    "snr_lo": float(edges[b]),
                    "snr_hi": float(edges[b + 1]),
                    "n": nb,
                    "bler": float(blers[m].mean()),
                    "ber": float(bers[m].mean()),
                    "realized_flops_ratio": float(flops_per_sample[m].mean() / max_flops),
                }
            )

    return {
        "n": n_actual,
        "ber_mean": float(bers.mean()),
        "bler": float(blers.mean()),
        "realized_flops_mean": float(flops_per_sample.mean()),
        "realized_flops_ratio": float(flops_per_sample.mean() / max_flops),
        "max_flops": max_flops,
        "base_flops": base_flops,
        "routing_pct": {name: routing_counts[name] / n_actual for name in EXPERT_NAMES},
        "sink_redirect_pct": routing_counts["sink_redirect"] / n_actual,
        "per_snr": per_snr,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data-dir", type=Path, required=True, help="dir with uma.pt and tdlc.pt")
    ap.add_argument("--out-json", type=Path, default=PROJECT_ROOT / "docs" / "figures" / "inference_mask_results.json")
    ap.add_argument("--max-samples", type=int, default=32768, help="per profile")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ckpt] downloading {args.checkpoint}")
        ckpt_path = _download_artifact(args.checkpoint)
    print(f"[ckpt] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model_from_config(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    results = {}
    for profile in ["uma", "tdlc"]:
        print(f"\n=== {profile} ===")
        results[profile] = {}
        for mode in ["baseline", "A_mask", "B_sink"]:
            loader = build_cached_dataloader(
                path=args.data_dir / f"{profile}.pt",
                batch_size=128,
                num_workers=0,
            )
            r = evaluate_with_mode(model, loader, device, mode, args.max_samples)
            results[profile][mode] = r
            print(
                f"  [{mode}] BLER={r['bler']:.4f}  BER={r['ber_mean']:.4f}  "
                f"FLOPs%={r['realized_flops_ratio']*100:.1f}  "
                f"routing={r['routing_pct']}"
            )

    # Aggregate over both profiles
    avg = {}
    for mode in ["baseline", "A_mask", "B_sink"]:
        ublers = [results[p][mode]["bler"] for p in ["uma", "tdlc"]]
        ubers = [results[p][mode]["ber_mean"] for p in ["uma", "tdlc"]]
        ufratios = [results[p][mode]["realized_flops_ratio"] for p in ["uma", "tdlc"]]
        avg[mode] = {
            "avg_bler": float(np.mean(ublers)),
            "avg_ber": float(np.mean(ubers)),
            "avg_flops_ratio": float(np.mean(ufratios)),
        }
    results["average_over_profiles"] = avg

    print("\n=== Summary (avg over profiles) ===")
    for mode, v in avg.items():
        print(f"  {mode}: BLER={v['avg_bler']:.4f}  BER={v['avg_ber']:.4f}  " f"FLOPs={v['avg_flops_ratio']*100:.1f}%")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n[done] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
