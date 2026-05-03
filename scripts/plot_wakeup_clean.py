"""Clean asym-warm wakeup figure with annotations.

Uses W&B history for exp26 (t6lkdep2) to plot expert usage trajectories with
the "large crashes" / "large re-emerges" annotations highlighting the recipe
mechanism.

Output: docs/figures/report_wakeup.{pdf,png}
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
RUN = "t6lkdep2"  # exp26 asym-warm s67


def main() -> int:
    apply_style(font_size=10)
    out = PROJECT_ROOT / "docs" / "figures"

    print("[INFO] fetching exp26 history...", file=sys.stderr)
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{RUN}")
    keys = ["train/ema/expert_usage/nano",
            "train/ema/expert_usage/small",
            "train/ema/expert_usage/large"]
    df = run.history(keys=keys, samples=300, pandas=True).dropna()

    s = df["_step"].to_numpy()
    nano = df["train/ema/expert_usage/nano"].to_numpy()
    small = df["train/ema/expert_usage/small"].to_numpy()
    large = df["train/ema/expert_usage/large"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.0, 3.5))

    ax.plot(s, large, "-", color=C.LARGE, label="large (random init)", linewidth=2.4)
    ax.plot(s, nano, "-", color=C.NANO, label="nano (warm-started)", linewidth=1.8)
    ax.plot(s, small, "-", color=C.SMALL, label="small (warm-started)", linewidth=1.8)

    # Annotate the two regimes that matter for the recipe story.
    ax.axvspan(0, 1500, alpha=0.07, color=C.LARGE, zorder=0)
    ax.axvspan(3000, max(s[-1], 12000), alpha=0.07, color=C.SMALL, zorder=0)
    # Place region labels in the upper area where there is no line overlap.
    ax.text(750, 1.07, "cold large transiently dominates",
            fontsize=8.5, color=C.LARGE, ha="center", va="bottom", fontweight="bold")
    ax.text(7500, 1.07, "stable heterogeneous routing",
            fontsize=8.5, color=C.SMALL, ha="center", va="bottom", fontweight="bold")
    ax.legend(loc="center right", framealpha=0.95, fontsize=9, bbox_to_anchor=(0.98, 0.55))

    ax.set_xlabel("Training step")
    ax.set_ylabel("Routing share (EMA)")
    ax.set_ylim(-0.02, 1.18)
    ax.set_xlim(0, max(s[-1], 12000))
    ax.grid(True, axis="y")

    fig.savefig(out / "report_wakeup.pdf")
    fig.savefig(out / "report_wakeup.png")
    print(f"[OK] wrote {out}/report_wakeup.{{pdf,png}}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
