"""Generate poster-grade figures for the report and the poster.

All numerical values are verified against W&B runs and analysis JSONs as
documented in the report. This script is the single source of truth for
the three regenerated figures.

Outputs (PDF + PNG) into the same directory as this script:
  pareto_bler_flops.pdf       -- hero compute-vs-quality plot
  per_expert_success.pdf      -- bar chart, only large ever decodes
  per_snr_routing.pdf         -- TDL-C routing share by SNR bin
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DOCS_FIG = HERE.parent.parent / "docs" / "figures"

# ---- Style: FIT teal + warm orange accent + grayscale classical ------------
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 11,
        "legend.framealpha": 0.95,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    }
)

# Wong color-blind safe palette (https://www.nature.com/articles/nmeth.1618).
# Cool blues for the dense ladder, green for classical reference, warm pink/red for ours.
C_DENSE_NANO = "#56B4E9"  # sky blue
C_DENSE_SMALL = "#0072B2"  # blue
C_DENSE_LARGE = "#003D60"  # dark blue
C_OURS_BASE = "#CC79A7"  # reddish purple (muted hero)
C_OURS_B = "#D55E00"  # vermilion (saturated hero)
C_LMMSE = "#009E73"  # bluish green
C_CLASSICAL = "#009E73"


# ============================================================================
# 1. Pareto plot: BLER vs FLOPs
# ============================================================================
def pareto_plot():
    """Hero plot. Avg BLER on y, avg FLOPs (% of dense_large) on x.

    Verified data (all from W&B test eval runs and lmmse_snr20_results.json):
      dense_nano  (W&B bx7hylp6): UMa 0.961 / TDLC 0.941 -> avg 0.951 at 320M (20%)
      dense_small (W&B 8haq7zuz): UMa 0.951 / TDLC 0.911 -> avg 0.931 at 695M (43%)
      dense_large (W&B gpmfhn6k): UMa 0.936 / TDLC 0.866 -> avg 0.901 at 1604M (100%)
      exp26 base  (W&B 2zboo1rh): UMa 0.937 / TDLC 0.867 -> avg 0.9021 at 55.8%
      exp26 + B (inference_mask_exp26.json):       avg 0.9021 at 47.3%
      LMMSE LS-MRC (lmmse_snr20_results.json):     avg 0.8997 (no neural FLOPs)
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    # Dense neural ladder (blue gradient)
    dense = [
        ("dense\\_nano", 20.0, 0.951, C_DENSE_NANO),
        ("dense\\_small", 43.3, 0.931, C_DENSE_SMALL),
        ("dense\\_large", 100.0, 0.901, C_DENSE_LARGE),
    ]
    for name, x, y, c in dense:
        ax.scatter(x, y, marker="s", s=140, color=c, zorder=5, edgecolors="white", linewidths=1.2)
        ax.annotate(
            name,
            (x, y),
            xytext=(8, -8) if "large" in name else (8, 8),
            textcoords="offset points",
            fontsize=11,
            color=c,
            ha="left" if "large" not in name else "right",
            va="top" if "large" in name else "bottom",
        )

    # exp26 baseline (trained MoE, hard top-1)
    ax.scatter(55.8, 0.9021, marker="o", s=160, color=C_OURS_BASE, zorder=8, edgecolors="white", linewidths=1.5)
    ax.annotate(
        "exp26 (trained MoE)",
        (55.8, 0.9021),
        xytext=(8, -8),
        textcoords="offset points",
        fontsize=11,
        color=C_OURS_BASE,
        ha="left",
        va="top",
    )

    # exp26 + Mode B (HERO POINT)
    ax.scatter(47.3, 0.9021, marker="*", s=520, color=C_OURS_B, zorder=12, edgecolors="white", linewidths=1.8)
    ax.annotate(
        "exp26 + Mode B\n(ours, no retraining)",
        (47.3, 0.9021),
        xytext=(-15, 22),
        textcoords="offset points",
        fontsize=12,
        color=C_OURS_B,
        fontweight="bold",
        ha="right",
        va="bottom",
        arrowprops=dict(arrowstyle="->", color=C_OURS_B, lw=1.2),
    )

    # LMMSE classical reference line (math, not neural FLOPs)
    ax.axhline(
        0.8997,
        color=C_LMMSE,
        linestyle=":",
        linewidth=1.5,
        zorder=2,
        label="LMMSE LS-MRC (classical, no neural compute)",
    )

    # Pareto frontier through the four useful neural points (dense ladder + Mode B)
    pareto_pts = sorted(
        [
            (20.0, 0.951),
            (43.3, 0.931),
            (47.3, 0.9021),
            (100.0, 0.901),
        ]
    )
    px, py = zip(*pareto_pts)
    ax.plot(px, py, "-", color=C_OURS_B, alpha=0.20, linewidth=2.5, zorder=3)

    ax.set_xlabel("Average realised FLOPs (\\% of dense\\_large)")
    ax.set_ylabel("Average BLER (lower is better)")
    ax.set_xlim(10, 110)
    ax.set_ylim(0.885, 0.965)
    ax.invert_yaxis()
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True)

    fig.savefig(HERE / "pareto_bler_flops.pdf")
    fig.savefig(HERE / "pareto_bler_flops.png")
    print("Saved pareto_bler_flops")
    plt.close()


# ============================================================================
# 2. Per-expert block success rate
# ============================================================================
def per_expert_success():
    """Bar chart from router_mechanism_success_rate.json.

    Verified values:
      UMa  nano: 0.0%, small: 0.0%, large: 23.21%
      TDLC nano: 0.0%, small: 0.0%, large: 29.42%
    """
    sr_path = DOCS_FIG / "router_mechanism_success_rate.json"
    data = json.loads(sr_path.read_text())
    rates = {(r["profile"], r["expert"]): r["agg_success_rate_pct"] for r in data}

    experts = ["nano", "small", "large"]
    uma = [rates[("uma", e)] for e in experts]
    tdlc = [rates[("tdlc", e)] for e in experts]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.arange(len(experts))
    w = 0.36
    bars_u = ax.bar(x - w / 2, uma, w, color=C_DENSE_SMALL, label="UMa", edgecolor="white")
    bars_t = ax.bar(x + w / 2, tdlc, w, color=C_OURS_B, label="TDL-C", edgecolor="white")

    for bar, val in zip(bars_u, uma):
        h = bar.get_height()
        ax.annotate(
            f"{val:.1f}\\%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            color=C_DENSE_SMALL,
            fontweight="bold",
        )
    for bar, val in zip(bars_t, tdlc):
        h = bar.get_height()
        ax.annotate(
            f"{val:.1f}\\%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            color=C_OURS_B,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(experts)
    ax.set_ylabel("Block-success rate on routed samples (\\%)")
    ax.set_ylim(0, 35)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(True, axis="y")

    ax.text(
        0.5,
        -0.18,
        "Only \\textbf{large} ever decodes a block. nano and small are pure compute-skip mechanisms.",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        style="italic",
        color="#444",
    )

    fig.savefig(HERE / "per_expert_success.pdf")
    fig.savefig(HERE / "per_expert_success.png")
    print("Saved per_expert_success")
    plt.close()


# ============================================================================
# 3. Per-SNR routing share on TDL-C (stacked bar)
# ============================================================================
def per_snr_routing():
    """Stacked bar of routing share by SNR bin on TDL-C, exp26.

    Hard-coded from W&B eval `2zboo1rh` `eval/expert_usage_snr_binned.table.json`
    (7-bin standard eval; verified inline).
    Bins are 7 equal SNR bins from -10 to 25 dB on TDL-C.
    """
    # Verified from W&B 2zboo1rh expert_usage_snr_binned.table.json (TDL-C only).
    snr_centers = [-7.85, -3.57, 0.72, 5.00, 9.29, 13.57, 17.86]
    nano = [0.7628, 0.2537, 0.0242, 0.0004, 0.0000, 0.0000, 0.0000]
    small = [0.2372, 0.7463, 0.9738, 0.7547, 0.1282, 0.0075, 0.0009]
    large = [0.0000, 0.0000, 0.0020, 0.2448, 0.8718, 0.9925, 0.9991]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(snr_centers))
    w = 0.7
    p1 = ax.bar(x, nano, w, color=C_DENSE_NANO, edgecolor="white", label="nano")
    p2 = ax.bar(x, small, w, bottom=nano, color=C_DENSE_SMALL, edgecolor="white", label="small")
    p3 = ax.bar(
        x, large, w, bottom=np.array(nano) + np.array(small), color=C_DENSE_LARGE, edgecolor="white", label="large"
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:+.0f}" for s in snr_centers])
    ax.set_xlabel("SNR (dB), TDL-C")
    ax.set_ylabel("Router share")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    ax.grid(True, axis="y")

    fig.savefig(HERE / "per_snr_routing.pdf")
    fig.savefig(HERE / "per_snr_routing.png")
    print("Saved per_snr_routing")
    plt.close()


if __name__ == "__main__":
    pareto_plot()
    per_expert_success()
    per_snr_routing()
