"""Analyze static and SNR-oracle cascade baselines vs the learned MoE router.

Pulls per-SNR-bin BLER from existing W&B eval runs (dense_{nano,small,large}
+ exp26 MoE) and computes:
  - Always-X overall BLER + FLOPs (per-expert reference)
  - Per-profile static (best dense per profile)
  - SNR-oracle cascade: per SNR bin, pick cheapest expert that achieves
    BLER within `--quality-tolerance` of dense_large's BLER for that bin.
  - Comparison vs the learned MoE router (exp26)

Output: markdown table written to stdout (or --out path).

No GPU needed. Run locally with `uv run python scripts/analyze_static_baselines.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import wandb

# Per-expert FLOPs (from CLAUDE.md MoE Architecture table)
EXPERT_FLOPS = {
    "nano": 320e6,
    "small": 695e6,
    "large": 1604e6,
}
DENSE_LARGE_FLOPS = EXPERT_FLOPS["large"]

# Source eval runs (W&B run IDs). Update if newer evals supersede these.
DEFAULT_RUNS = {
    # dense baselines (test eval against uma + tdlc)
    "dense_nano": "f4qeu7o3",
    "dense_small": "u1bolt4j",
    "dense_large": "87ki5ylk",
    # MoE winner (test eval against uma + tdlc)
    "exp26_a2e3": "2zboo1rh",
}

PROFILES = ("uma", "tdlc")


def _fetch_run_summary(api: wandb.Api, entity: str, project: str, run_id: str) -> dict[str, Any]:
    run = api.run(f"{entity}/{project}/{run_id}")
    return dict(run.summary)


def _extract_per_snr_bler(summary: dict[str, Any], profile: str, num_bins: int = 7) -> list[tuple[float, float]]:
    """Return list of (snr_center, bler) tuples for the eval profile."""
    out: list[tuple[float, float]] = []
    for i in range(num_bins):
        bler_key = f"eval/{profile}/snr_bin_{i}/bler"
        snr_key = f"eval/{profile}/snr_bin_{i}/snr_center"
        if bler_key not in summary or snr_key not in summary:
            break
        out.append((float(summary[snr_key]), float(summary[bler_key])))
    if not out:
        # fall back to val keys (some runs only have val)
        for i in range(num_bins):
            bler_key = f"val/{profile}/snr_bin_{i}/bler"
            snr_key = f"val/{profile}/snr_bin_{i}/snr_center"
            if bler_key not in summary or snr_key not in summary:
                break
            out.append((float(summary[snr_key]), float(summary[bler_key])))
    return out


def _extract_overall_bler_flops(summary: dict[str, Any], profile: str) -> tuple[float, float]:
    bler = float(summary.get(f"eval/{profile}/bler", float("nan")))
    realized = float(summary.get(f"eval/{profile}/realized_flops", float("nan")))
    return bler, realized


def _compute_per_profile_static(
    dense_per_snr: dict[str, dict[str, list[tuple[float, float]]]],
) -> dict[str, dict[str, Any]]:
    """For each profile, pick the cheapest expert that's within 1pp of dense_large's overall BLER."""
    out: dict[str, dict[str, Any]] = {}
    for profile in PROFILES:
        # use overall BLER per expert
        overall = {
            name: sum(b for _, b in dense_per_snr[name][profile]) / len(dense_per_snr[name][profile])
            for name in ("nano", "small", "large")
        }
        large_bler = overall["large"]
        # ranked cheapest → most expensive
        for name in ("nano", "small", "large"):
            if overall[name] - large_bler < 0.01:  # within 1pp
                out[profile] = {"expert": name, "bler": overall[name], "flops": EXPERT_FLOPS[name]}
                break
        else:
            out[profile] = {"expert": "large", "bler": overall["large"], "flops": EXPERT_FLOPS["large"]}
    return out


def _compute_snr_oracle_cascade(
    dense_per_snr: dict[str, dict[str, list[tuple[float, float]]]],
    quality_tolerance: float,
) -> dict[str, dict[str, Any]]:
    """For each (profile, SNR bin), pick the cheapest expert whose BLER is within
    `quality_tolerance` of dense_large's BLER for that bin."""
    result: dict[str, dict[str, Any]] = {}
    for profile in PROFILES:
        bins_per_expert = {name: dense_per_snr[name][profile] for name in ("nano", "small", "large")}
        n_bins = len(bins_per_expert["large"])
        chosen_per_bin: list[tuple[float, str, float]] = []  # (snr, expert, bler)
        for i in range(n_bins):
            snr, large_bler = bins_per_expert["large"][i]
            for name in ("nano", "small", "large"):
                if i >= len(bins_per_expert[name]):
                    continue
                _, bler = bins_per_expert[name][i]
                if bler - large_bler < quality_tolerance:
                    chosen_per_bin.append((snr, name, bler))
                    break
            else:
                chosen_per_bin.append((snr, "large", large_bler))
        # uniform-weighted average across bins
        avg_bler = sum(b for _, _, b in chosen_per_bin) / len(chosen_per_bin)
        avg_flops = sum(EXPERT_FLOPS[e] for _, e, _ in chosen_per_bin) / len(chosen_per_bin)
        result[profile] = {
            "bler": avg_bler,
            "flops": avg_flops,
            "per_bin": chosen_per_bin,
        }
    return result


def _format_md_table(rows: list[dict[str, Any]]) -> str:
    headers = ["Strategy", "TDLC BLER", "UMA BLER", "Avg BLER", "Avg FLOPs (M)", "Avg FLOPs %", "Notes"]
    lines = ["| " + " | ".join(headers) + " |", "|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    f"{row['tdlc_bler']:.3f}",
                    f"{row['uma_bler']:.3f}",
                    f"{(row['tdlc_bler'] + row['uma_bler']) / 2:.3f}",
                    f"{(row['tdlc_flops'] + row['uma_flops']) / 2 / 1e6:.0f}",
                    f"{(row['tdlc_flops'] + row['uma_flops']) / 2 / DENSE_LARGE_FLOPS * 100:.0f}%",
                    row.get("notes", ""),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="knn_moe-5g-nrx")
    parser.add_argument("--project", default="moe-5g-nrx")
    parser.add_argument(
        "--quality-tolerance",
        type=float,
        default=0.05,
        help="SNR-oracle cascade picks cheapest expert with BLER within this tolerance of large's BLER per bin",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write markdown table to this file")
    parser.add_argument("--json", type=Path, default=None, help="Also dump raw analysis as JSON")
    args = parser.parse_args()

    api = wandb.Api()

    print("[INFO] Fetching W&B run summaries...", file=sys.stderr)
    summaries = {}
    for label, run_id in DEFAULT_RUNS.items():
        print(f"  - {label} ({run_id})", file=sys.stderr)
        summaries[label] = _fetch_run_summary(api, args.entity, args.project, run_id)

    # Extract per-SNR BLER for dense experts
    dense_per_snr = {
        name: {profile: _extract_per_snr_bler(summaries[f"dense_{name}"], profile) for profile in PROFILES}
        for name in ("nano", "small", "large")
    }

    # Overall numbers for dense + MoE winner
    rows: list[dict[str, Any]] = []
    for name in ("nano", "small", "large"):
        s = summaries[f"dense_{name}"]
        tdlc_bler, tdlc_flops = _extract_overall_bler_flops(s, "tdlc")
        uma_bler, uma_flops = _extract_overall_bler_flops(s, "uma")
        # Dense FLOPs is fixed per expert
        rows.append(
            {
                "name": f"Always-{name}",
                "tdlc_bler": tdlc_bler,
                "tdlc_flops": EXPERT_FLOPS[name],
                "uma_bler": uma_bler,
                "uma_flops": EXPERT_FLOPS[name],
                "notes": "fixed dense baseline",
            }
        )

    # Per-profile static
    static = _compute_per_profile_static(dense_per_snr)
    rows.append(
        {
            "name": "Per-profile static",
            "tdlc_bler": static["tdlc"]["bler"],
            "tdlc_flops": static["tdlc"]["flops"],
            "uma_bler": static["uma"]["bler"],
            "uma_flops": static["uma"]["flops"],
            "notes": f"TDLC→{static['tdlc']['expert']}, UMA→{static['uma']['expert']}",
        }
    )

    # SNR-oracle cascade (with quality tolerance)
    cascade = _compute_snr_oracle_cascade(dense_per_snr, args.quality_tolerance)
    rows.append(
        {
            "name": f"SNR-oracle cascade (≤+{args.quality_tolerance:.2f} pp BLER vs large per bin)",
            "tdlc_bler": cascade["tdlc"]["bler"],
            "tdlc_flops": cascade["tdlc"]["flops"],
            "uma_bler": cascade["uma"]["bler"],
            "uma_flops": cascade["uma"]["flops"],
            "notes": "uses TRUE per-sample SNR (oracle, not deployable)",
        }
    )

    # MoE winner
    s_moe = summaries["exp26_a2e3"]
    tdlc_bler, tdlc_flops = _extract_overall_bler_flops(s_moe, "tdlc")
    uma_bler, uma_flops = _extract_overall_bler_flops(s_moe, "uma")
    rows.append(
        {
            "name": "**Learned MoE (exp26)**",
            "tdlc_bler": tdlc_bler,
            "tdlc_flops": tdlc_flops,
            "uma_bler": uma_bler,
            "uma_flops": uma_flops,
            "notes": "α=2e-3 winner, channel-aware routing",
        }
    )

    table = _format_md_table(rows)
    print(table)
    if args.out is not None:
        args.out.write_text(table + "\n", encoding="utf-8")
        print(f"\n[INFO] Wrote table to {args.out}", file=sys.stderr)
    if args.json is not None:
        payload = {
            "rows": rows,
            "snr_oracle_per_bin": {p: cascade[p]["per_bin"] for p in PROFILES},
            "quality_tolerance": args.quality_tolerance,
            "source_runs": DEFAULT_RUNS,
        }
        args.json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"[INFO] Wrote raw data to {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
