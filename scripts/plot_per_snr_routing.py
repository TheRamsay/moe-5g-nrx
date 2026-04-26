"""Plot per-SNR-bin expert routing for the exp26 winner (and optionally peers).

Pulls `eval/expert_usage_snr_binned` table from W&B eval runs and makes a
stacked bar plot of routing share per SNR bin per profile. No GPU needed.

Usage: `uv run python scripts/plot_per_snr_routing.py --out figures/`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import wandb

ENTITY = "knn_moe-5g-nrx"
PROJECT = "moe-5g-nrx"

# Eval runs to plot — expand if needed
RUNS = {
    "exp26 (α=2e-3 winner)": "2zboo1rh",
    # "exp25 (α=1e-3)": "5jswm490",
    # "exp27 (α=5e-3)": "dh4x0qmu",
}

EXPERT_COLORS = {"nano": "#4daf4a", "small": "#377eb8", "large": "#e41a1c"}
PROFILES = ("uma", "tdlc")


def load_routing(api: wandb.Api, run_id: str) -> dict[str, list[dict]]:
    """Return {profile: [{snr_center, expert, usage}, ...]} from the W&B table artifact."""
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    out: dict[str, list[dict]] = {p: [] for p in PROFILES}
    for art in run.logged_artifacts():
        if "expert_usage_snr_binned" not in art.name:
            continue
        art_dir = art.download(root=f"/tmp/wandb_routing/{run_id}/{art.name.replace(':', '_')}")
        for table_path in Path(art_dir).rglob("*.table.json"):
            data = json.loads(Path(table_path).read_text())
            cols = data["columns"]
            rows = data["data"]
            try:
                p_idx = cols.index("profile")
                snr_idx = cols.index("snr_center")
                exp_idx = cols.index("expert")
                usage_idx = cols.index("usage")
            except ValueError:
                continue
            for row in rows:
                profile = str(row[p_idx]).lower()
                if profile not in out:
                    continue
                out[profile].append(
                    {
                        "snr_center": float(row[snr_idx]),
                        "expert": str(row[exp_idx]),
                        "usage": float(row[usage_idx]),
                    }
                )
            return out
    return out


def plot_routing(routing: dict[str, list[dict]], title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(PROFILES), figsize=(11, 4), sharey=True)
    for ax, profile in zip(axes, PROFILES):
        rows = routing[profile]
        if not rows:
            ax.set_title(f"{profile.upper()}: no data")
            continue
        snr_centers = sorted({r["snr_center"] for r in rows})
        experts = ["nano", "small", "large"]
        usage = {e: [0.0] * len(snr_centers) for e in experts}
        for r in rows:
            try:
                i = snr_centers.index(r["snr_center"])
            except ValueError:
                continue
            if r["expert"] in usage:
                usage[r["expert"]][i] = r["usage"]
        x = np.arange(len(snr_centers))
        bottom = np.zeros(len(snr_centers))
        for e in experts:
            vals = np.array(usage[e])
            ax.bar(x, vals, bottom=bottom, color=EXPERT_COLORS[e], label=e, edgecolor="white", linewidth=0.5)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:.0f}" for s in snr_centers])
        ax.set_xlabel("SNR (dB, bin center)")
        ax.set_title(profile.upper())
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Expert usage (fraction)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Expert")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[INFO] wrote {out_path}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    for label, run_id in RUNS.items():
        routing = load_routing(api, run_id)
        out_png = args.out / f"per_snr_routing_{run_id}.png"
        plot_routing(routing, label, out_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
