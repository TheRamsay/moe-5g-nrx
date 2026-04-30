"""Plot routing trajectories over training steps for collapse-dynamics analysis.

For each run, fetches the EMA-smoothed expert usage (nano/small/large) and the
router entropy over training steps, then plots a comparison figure showing
WHEN each training paradigm commits its routing decisions.

Output: docs/figures/routing_trajectories_{collapse_modes,bimodal_seeds,
        anti_collapse}.png (3 panels per figure).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import wandb

ENTITY = "knn_moe-5g-nrx"
PROJECT = "moe-5g-nrx"

KEYS = [
    "train/ema/expert_usage/nano",
    "train/ema/expert_usage/small",
    "train/ema/expert_usage/large",
    "train/ema/router_entropy",
    "train/ema/realized_flops_ratio",
]


def fetch_history(run_id: str, samples: int = 500) -> dict[str, np.ndarray]:
    """Pull sampled history for one run."""
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    df = run.history(keys=KEYS, samples=samples, pandas=True)
    if df.empty:
        raise RuntimeError(f"no history for run {run_id}")
    out = {"steps": df["_step"].to_numpy(), "name": run.name}
    for k in KEYS:
        out[k] = df[k].to_numpy() if k in df.columns else np.full(len(df), np.nan)
    return out


# --- the runs we care about ---------------------------------------------------

PHASE1 = ("2op33pak", "Phase 1 (cold start)")
PHASE2_V1 = ("89no8f1k", "Phase 2 v1 (full warm)")
ASYM_S67 = ("t6lkdep2", "Asym warm s67 (exp26 — winner)")
ASYM_S42 = ("121ex9e6", "Asym warm s42 (exp29 — works)")

SWITCH_AUX_W1E_3 = ("j6vwy0hu", "Switch aux w=1e-3")
SWITCH_AUX_W1E_2 = ("vst322r9", "Switch aux w=1e-2")
SWITCH_AUX_W1E_1 = ("gfmiqhai", "Switch aux w=1e-1")
SWITCH_AUX_W1E0 = ("q2m2yr65", "Switch aux w=1e0")
CAPACITY_W0P1 = ("vpvaiqnv", "Capacity w=0.1")
CAPACITY_W0P5 = ("be3tvpqf", "Capacity w=0.5")
CAPACITY_W2P0 = ("1e63z076", "Capacity w=2.0")


def stack_routing(h: dict) -> np.ndarray:
    """Return (T, 3) array: nano / small / large EMA usage."""
    return np.stack(
        [
            h["train/ema/expert_usage/nano"],
            h["train/ema/expert_usage/small"],
            h["train/ema/expert_usage/large"],
        ],
        axis=1,
    )


EXPERT_COLORS = {"nano": "#fdc086", "small": "#7fc97f", "large": "#beaed4"}


def plot_routing_panel(ax, h, title, with_legend=False):
    routing = stack_routing(h)
    # stacked-area plot: nano (bottom), small (middle), large (top)
    ax.fill_between(h["steps"], 0, routing[:, 0], color=EXPERT_COLORS["nano"], alpha=0.85, label="nano")
    ax.fill_between(
        h["steps"],
        routing[:, 0],
        routing[:, 0] + routing[:, 1],
        color=EXPERT_COLORS["small"],
        alpha=0.85,
        label="small",
    )
    ax.fill_between(
        h["steps"],
        routing[:, 0] + routing[:, 1],
        routing[:, 0] + routing[:, 1] + routing[:, 2],
        color=EXPERT_COLORS["large"],
        alpha=0.85,
        label="large",
    )
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(h["steps"][-1], 1))
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("EMA expert usage")
    ax.grid(True, alpha=0.25)
    if with_legend:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)


def plot_entropy_panel(ax, runs_with_labels):
    for h, label in runs_with_labels:
        ax.plot(h["steps"], h["train/ema/router_entropy"], label=label, linewidth=1.5)
    ax.axhline(y=np.log(3), color="gray", linestyle="--", alpha=0.5, label="max entropy log(3) ≈ 1.10")
    ax.set_ylabel("Router entropy (EMA)")
    ax.set_xlabel("Training step")
    ax.set_title("Router entropy over training (collapse diagnostic)", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)


def figure_collapse_modes(out_dir: Path):
    """The 3-paradigm comparison: Phase 1 / Phase 2 v1 / Asym warm."""
    print("[INFO] Fetching: Phase 1, Phase 2 v1, Asym warm s67...", file=sys.stderr)
    p1 = fetch_history(PHASE1[0])
    p2 = fetch_history(PHASE2_V1[0])
    asym = fetch_history(ASYM_S67[0])

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    plot_routing_panel(axes[0], p1, PHASE1[1], with_legend=True)
    plot_routing_panel(axes[1], p2, PHASE2_V1[1])
    plot_routing_panel(axes[2], asym, ASYM_S67[1])
    plot_entropy_panel(axes[3], [(p1, "Phase 1"), (p2, "Phase 2 v1"), (asym, "Asym warm s67")])
    fig.suptitle("Three training paradigms — when does the router commit?", fontsize=13, y=1.0, fontweight="bold")
    plt.tight_layout()
    out = out_dir / "routing_trajectories_collapse_modes.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[INFO] wrote {out}", file=sys.stderr)
    plt.close(fig)


def figure_bimodal_seeds(out_dir: Path):
    """exp26 (s67) vs exp29 (s42) — both worked. Add comparison."""
    print("[INFO] Fetching: Asym warm s67 + s42...", file=sys.stderr)
    s67 = fetch_history(ASYM_S67[0])
    s42 = fetch_history(ASYM_S42[0])

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    plot_routing_panel(axes[0], s67, ASYM_S67[1], with_legend=True)
    plot_routing_panel(axes[1], s42, ASYM_S42[1])
    plot_entropy_panel(axes[2], [(s67, "s67 (winner)"), (s42, "s42")])
    fig.suptitle(
        "Asym warm-start: 2-of-3 working seeds — divergent trajectories early", fontsize=13, y=1.0, fontweight="bold"
    )
    plt.tight_layout()
    out = out_dir / "routing_trajectories_bimodal_seeds.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[INFO] wrote {out}", file=sys.stderr)
    plt.close(fig)


def figure_anti_collapse(out_dir: Path):
    """Phase 2 v1 + 4 switch aux + 3 capacity sweeps — all collapse different ways."""
    print("[INFO] Fetching: Phase 2 v1 + 4 switch aux + 3 capacity sweeps...", file=sys.stderr)
    runs = {
        "Phase 2 v1": fetch_history(PHASE2_V1[0]),
        "Switch aux w=1e-3": fetch_history(SWITCH_AUX_W1E_3[0]),
        "Switch aux w=1e-2": fetch_history(SWITCH_AUX_W1E_2[0]),
        "Switch aux w=1e-1": fetch_history(SWITCH_AUX_W1E_1[0]),
        "Switch aux w=1e0": fetch_history(SWITCH_AUX_W1E0[0]),
        "Capacity w=0.1": fetch_history(CAPACITY_W0P1[0]),
        "Capacity w=0.5": fetch_history(CAPACITY_W0P5[0]),
        "Capacity w=2.0": fetch_history(CAPACITY_W2P0[0]),
    }

    # 3x3 grid: 8 routing panels + 1 entropy panel
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    axes_flat = axes.flatten()
    items = list(runs.items())
    for ax, (label, h) in zip(axes_flat, items):
        plot_routing_panel(ax, h, label, with_legend=(label == "Phase 2 v1"))
    plot_entropy_panel(axes_flat[8], list((h, label) for label, h in runs.items()))
    fig.suptitle(
        "Anti-collapse mechanisms vs Phase 2 baseline — none recover heterogeneous routing",
        fontsize=13,
        y=1.0,
        fontweight="bold",
    )
    plt.tight_layout()
    out = out_dir / "routing_trajectories_anti_collapse.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[INFO] wrote {out}", file=sys.stderr)
    plt.close(fig)


def main():
    out_dir = Path(__file__).resolve().parents[1] / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_collapse_modes(out_dir)
    figure_bimodal_seeds(out_dir)
    figure_anti_collapse(out_dir)
    print("[DONE]", file=sys.stderr)


if __name__ == "__main__":
    main()
