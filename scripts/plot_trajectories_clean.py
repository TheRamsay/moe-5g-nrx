"""Clean three-paradigm routing trajectories figure for the report.

Pulls EMA-smoothed expert usage and router entropy for the three training
paradigms (Phase 1 cold, Phase 2 full warm, asym warm s67 = exp26) from W&B
and renders a 2x2 grid with consistent palette, no em-dashes, and clear
visual hierarchy.

Output: docs/figures/report_trajectories.{pdf,png}

Verified W&B run IDs:
  Phase 1 cold start:        2op33pak
  Phase 2 v1 full warm:      89no8f1k
  Asym warm s67 (exp26):     t6lkdep2
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts._figstyle import C, apply_style  # noqa: E402

ENTITY = "knn_moe-5g-nrx"
PROJECT = "moe-5g-nrx"

KEYS = [
    "train/ema/expert_usage/nano",
    "train/ema/expert_usage/small",
    "train/ema/expert_usage/large",
    "train/ema/router_entropy",
]

PHASE1 = ("2op33pak", "Phase 1 (cold start)")
PHASE2 = ("89no8f1k", "Phase 2 (full warm-start)")
ASYM = ("t6lkdep2", "Asymmetric warm-start, cold-large (exp26)")


def fetch(run_id: str, samples: int = 500) -> dict:
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    df = run.history(keys=KEYS, samples=samples, pandas=True)
    out = {"steps": df["_step"].to_numpy()}
    for k in KEYS:
        out[k] = df[k].to_numpy() if k in df.columns else np.full(len(df), np.nan)
    return out


def plot_routing(ax, h, title, with_legend=False):
    nano = h["train/ema/expert_usage/nano"]
    small = h["train/ema/expert_usage/small"]
    large = h["train/ema/expert_usage/large"]
    s = h["steps"]

    ax.fill_between(s, 0, nano, color=C.NANO, alpha=0.85, label="nano", linewidth=0)
    ax.fill_between(s, nano, nano + small, color=C.SMALL, alpha=0.85, label="small", linewidth=0)
    ax.fill_between(s, nano + small, nano + small + large, color=C.LARGE,
                    alpha=0.85, label="large", linewidth=0)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, 10000)
    ax.set_title(title, fontsize=10, fontweight="bold", color=C.AXIS_GREY)
    ax.set_ylabel("Routing share (EMA)")
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True, axis="y")
    if with_legend:
        ax.legend(loc="upper right", ncol=3, fontsize=8.5,
                  framealpha=0.95, columnspacing=1.0, handlelength=1.4)


def plot_entropy(ax, runs):
    colors = {"Phase 1": C.NEUTRAL_GREY, "Phase 2": C.HEADLINE, "Asym warm (exp26)": C.LARGE}
    for h, label in runs:
        ax.plot(h["steps"], h["train/ema/router_entropy"], label=label,
                linewidth=1.8, color=colors[label])
    ax.axhline(np.log(3), color="#888888", linestyle="--", linewidth=0.8,
               alpha=0.7, label=r"max entropy $\log 3 \approx 1.10$")
    ax.set_ylabel("Router entropy (EMA)")
    ax.set_xlabel("Training step")
    ax.set_title("Router entropy (collapse diagnostic)", fontsize=10,
                 fontweight="bold", color=C.AXIS_GREY)
    ax.set_ylim(0, 1.2)
    ax.set_xlim(0, 10000)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95, ncol=2,
              bbox_to_anchor=(1.0, 0.55))


def main() -> int:
    apply_style(font_size=10)
    out = PROJECT_ROOT / "docs" / "figures"

    print("[INFO] fetching three runs from W&B...", file=sys.stderr)
    p1 = fetch(PHASE1[0])
    p2 = fetch(PHASE2[0])
    asym = fetch(ASYM[0])

    fig, axes = plt.subplots(4, 1, figsize=(7.5, 8.0), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1, 1, 1.05], "hspace": 0.45})

    plot_routing(axes[0], p1, PHASE1[1], with_legend=True)
    plot_routing(axes[1], p2, PHASE2[1])
    plot_routing(axes[2], asym, ASYM[1])
    plot_entropy(axes[3], [(p1, "Phase 1"), (p2, "Phase 2"), (asym, "Asym warm (exp26)")])

    fig.savefig(out / "report_trajectories.pdf")
    fig.savefig(out / "report_trajectories.png")
    print(f"[OK] wrote {out}/report_trajectories.{{pdf,png}}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
