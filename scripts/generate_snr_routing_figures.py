"""Generate per-SNR routing analysis figures."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 8,
        "legend.framealpha": 0.9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    }
)

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

C_NANO = "#E69F00"
C_SMALL = "#0072B2"
C_LARGE = "#009E73"
C_RED = "#D55E00"


def expert_usage_by_snr():
    """Stacked bar chart: expert usage per SNR bin for asym warm 12k."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)

    # TDLC
    tdlc_snr = [-8, -4, 1, 5, 9, 14, 18]
    tdlc_nano = [0.159, 0.003, 0.0, 0.0, 0.0, 0.0, 0.0]
    tdlc_small = [0.841, 0.996, 0.904, 0.511, 0.225, 0.104, 0.069]
    tdlc_large = [0.0, 0.001, 0.096, 0.489, 0.775, 0.896, 0.931]

    # UMA
    uma_snr = [-3, 1, 6, 10, 14, 19, 23]
    uma_nano = [0.595, 0.491, 0.413, 0.316, 0.198, 0.103, 0.018]
    uma_small = [0.401, 0.424, 0.378, 0.387, 0.392, 0.394, 0.373]
    uma_large = [0.005, 0.085, 0.209, 0.297, 0.411, 0.503, 0.609]

    width = 3.5

    for ax, snr, nano, small, large, title in [
        (axes[0], tdlc_snr, tdlc_nano, tdlc_small, tdlc_large, "TDL-C"),
        (axes[1], uma_snr, uma_nano, uma_small, uma_large, "UMa"),
    ]:
        snr_arr = np.array(snr)
        ax.bar(snr_arr, nano, width, label="nano", color=C_NANO, edgecolor="white", linewidth=0.3)
        ax.bar(snr_arr, small, width, bottom=nano, label="small", color=C_SMALL, edgecolor="white", linewidth=0.3)
        ax.bar(
            snr_arr,
            large,
            width,
            bottom=np.array(nano) + np.array(small),
            label="large",
            color=C_LARGE,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.set_xlabel("SNR (dB)")
        ax.set_title(title)
        ax.set_xticks(snr)
        ax.set_xticklabels([str(s) for s in snr], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y")

    axes[0].set_ylabel("Expert Usage")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

    fig.suptitle("Asymmetric Warm-Start 12k: Expert Routing by SNR", fontsize=11, y=1.02)
    fig.savefig(OUT / "expert_usage_by_snr.png")
    fig.savefig(OUT / "expert_usage_by_snr.pdf")
    print("Saved expert_usage_by_snr")
    plt.close()


def bler_comparison_by_snr():
    """BLER vs SNR for asym warm vs dense large vs Phase 1, both profiles."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)

    # TDLC data
    tdlc_snr = [-8, -4, 1, 5, 9, 14, 18]

    tdlc_dense = [1.0, 1.0, 1.0, 1.0, 1.0, 0.753, 0.085]  # Phase 2 = dense large (collapsed)
    tdlc_phase1 = [1.0, 1.0, 1.0, 1.0, 1.0, 0.945, 0.391]
    tdlc_asym = [1.0, 1.0, 1.0, 1.0, 1.0, 0.892, 0.267]

    # UMA data
    uma_snr = [-3, 1, 6, 10, 14, 19, 23]

    uma_dense = [1.0, 1.0, 1.0, 1.0, 0.920, 0.809, 0.725]
    uma_phase1 = [1.0, 1.0, 1.0, 1.0, 0.962, 0.861, 0.788]
    uma_asym = [1.0, 1.0, 1.0, 1.0, 0.954, 0.848, 0.768]

    for ax, snr, dense, phase1, asym, title in [
        (axes[0], tdlc_snr, tdlc_dense, tdlc_phase1, tdlc_asym, "TDL-C"),
        (axes[1], uma_snr, uma_dense, uma_phase1, uma_asym, "UMa"),
    ]:
        ax.plot(snr, dense, "s-", color=C_RED, label="Dense large", markersize=5, linewidth=1.8)
        ax.plot(snr, asym, "D-", color=C_LARGE, label="Asym warm 12k", markersize=5, linewidth=1.8)
        ax.plot(snr, phase1, "^-", color=C_SMALL, label="Phase 1", markersize=5, linewidth=1.8)
        ax.set_xlabel("SNR (dB)")
        ax.set_title(title)
        ax.set_xticks(snr)
        ax.set_xticklabels([str(s) for s in snr], fontsize=8)
        ax.grid(True)

    axes[0].set_ylabel("BLER")
    axes[1].legend(loc="lower left")
    axes[0].set_ylim(-0.02, 1.08)

    fig.suptitle("BLER by SNR Bin: MoE vs Dense Baseline", fontsize=11, y=1.02)
    fig.savefig(OUT / "bler_by_snr_comparison.png")
    fig.savefig(OUT / "bler_by_snr_comparison.pdf")
    print("Saved bler_by_snr_comparison")
    plt.close()


if __name__ == "__main__":
    expert_usage_by_snr()
    bler_comparison_by_snr()
    print("Done!")
