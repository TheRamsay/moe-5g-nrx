"""Generate figures for the checkpoint report."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

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
C_GREY = "#999999"
C_RED = "#D55E00"
C_PURPLE = "#7B2D8E"
C_ROSE = "#CC79A7"


def pareto_plot():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    # Dense baselines
    dense = [
        ("Dense nano", 320, (0.971 + 0.961) / 2),
        ("Dense small", 695, (0.911 + 0.951) / 2),
        ("Dense large", 1604, (0.866 + 0.936) / 2),
    ]
    for name, flops, bler in dense:
        ax.scatter(flops, bler, marker="s", s=50, color=C_GREY, zorder=5)
        ha, dx = ("left", 12) if "large" not in name else ("right", -12)
        ax.annotate(
            name,
            (flops, bler),
            textcoords="offset points",
            xytext=(dx, 0),
            fontsize=7,
            color="#888888",
            ha=ha,
            va="center",
        )

    # MoE runs
    moe = [
        ("Phase 1 (joint)", (917 + 628) / 2, (0.906 + 0.945) / 2, C_SMALL, "^"),
        ("Phase 2 (collapsed)", 1604, (0.835 + 0.923) / 2, C_RED, "v"),
        ("Asym warm 6k", (601 + 639) / 2, (0.945 + 0.971) / 2, C_ROSE, "o"),
        ("Asym warm 12k", (1100 + 856) / 2, (0.881 + 0.939) / 2, C_LARGE, "D"),
        (r"$\beta$=2.0", (860 + 864) / 2, (0.962 + 0.975) / 2, C_PURPLE, "X"),
    ]
    for name, flops, bler, color, marker in moe:
        ax.scatter(flops, bler, marker=marker, s=70, color=color, zorder=10, edgecolors="white", linewidths=0.5)
        dx, dy, ha, va = 8, 5, "left", "bottom"
        if "Phase 2" in name:
            dx, dy, ha, va = -8, 5, "right", "bottom"
        elif "Phase 1" in name:
            dx, dy, ha, va = -8, 5, "right", "bottom"
        elif "6k" in name:
            dx, dy, ha, va = -8, 5, "right", "bottom"
        elif "12k" in name:
            dx, dy, ha, va = 8, -5, "left", "top"
        elif "beta" in name:
            dx, dy, ha, va = 8, -8, "left", "top"
        ax.annotate(
            name, (flops, bler), textcoords="offset points", xytext=(dx, dy), fontsize=7.5, color=color, ha=ha, va=va
        )

    # Pareto frontier
    pareto = sorted(
        [(1604, (0.835 + 0.923) / 2), ((1100 + 856) / 2, (0.881 + 0.939) / 2), ((917 + 628) / 2, (0.906 + 0.945) / 2)]
    )
    ax.plot(*zip(*pareto), "-", color=C_LARGE, alpha=0.25, linewidth=2, zorder=3)

    ax.set_xlabel("Average Realized FLOPs (M)")
    ax.set_ylabel("Average BLER (lower is better)")
    ax.set_xlim(250, 1750)
    ax.set_ylim(0.855, 0.985)
    ax.invert_yaxis()
    ax.grid(True)
    fig.savefig(OUT / "pareto_bler_flops.png")
    fig.savefig(OUT / "pareto_bler_flops.pdf")
    print("Saved pareto_bler_flops")
    plt.close()


def waterfall_plot():
    fig, ax = plt.subplots(figsize=(4.5, 3.3))

    snr = [13, 15, 17, 19]
    nano = [0.996, 0.922, 0.722, 0.452]
    small = [0.998, 0.834, 0.548, 0.310]
    large = [0.969, 0.649, 0.284, 0.105]

    ax.plot(snr, large, "D-", color=C_LARGE, label="large (450k)", markersize=5, linewidth=1.8)
    ax.plot(snr, small, "s-", color=C_SMALL, label="small (168k)", markersize=5, linewidth=1.8)
    ax.plot(snr, nano, "o-", color=C_NANO, label="nano (90k)", markersize=5, linewidth=1.8)

    # Gap annotation
    ax.annotate("", xy=(17, 0.29), xytext=(17, 0.72), arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1.2))
    ax.text(17.3, 0.50, "44 pp", fontsize=8.5, color=C_RED, fontweight="bold", va="center")

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BLER")
    ax.set_xticks(snr)
    ax.legend(loc="lower left")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True)
    fig.savefig(OUT / "waterfall_dense_baselines.png")
    fig.savefig(OUT / "waterfall_dense_baselines.pdf")
    print("Saved waterfall_dense_baselines")
    plt.close()


def expert_usage_plot():
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    # Original run (0-6k)
    s0 = [
        50,
        100,
        150,
        200,
        300,
        400,
        500,
        700,
        900,
        1000,
        1200,
        1400,
        1600,
        1800,
        2000,
        2050,
        2100,
        2400,
        2800,
        3000,
        3500,
        4000,
        4500,
        5000,
        5500,
        6000,
    ]
    l0 = [
        0.992,
        0.998,
        0.784,
        0.697,
        0.693,
        0.665,
        0.649,
        0.628,
        0.503,
        0.427,
        0.384,
        0.358,
        0.371,
        0.354,
        0.353,
        0.003,
        0.0001,
        0.00003,
        0.00002,
        0.00001,
        0.00001,
        0.00002,
        0.00001,
        0.00001,
        0.00007,
        0.00005,
    ]
    n0 = [
        0.007,
        0.002,
        0.216,
        0.303,
        0.307,
        0.335,
        0.351,
        0.371,
        0.215,
        0.225,
        0.233,
        0.255,
        0.266,
        0.267,
        0.263,
        0.418,
        0.433,
        0.421,
        0.430,
        0.428,
        0.447,
        0.456,
        0.467,
        0.464,
        0.462,
        0.466,
    ]
    sm0 = [
        0.001,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.001,
        0.283,
        0.347,
        0.382,
        0.386,
        0.363,
        0.379,
        0.384,
        0.579,
        0.566,
        0.579,
        0.569,
        0.572,
        0.553,
        0.544,
        0.533,
        0.536,
        0.538,
        0.534,
    ]

    # Resume run (6k-12k)
    s1 = [
        6050,
        6200,
        6400,
        6550,
        6600,
        6650,
        6700,
        7000,
        7500,
        8000,
        8500,
        9000,
        9500,
        10000,
        10500,
        11000,
        11500,
        12000,
    ]
    l1 = [
        0.00005,
        0.00008,
        0.00025,
        0.007,
        0.168,
        0.260,
        0.261,
        0.284,
        0.293,
        0.305,
        0.306,
        0.309,
        0.319,
        0.327,
        0.306,
        0.323,
        0.320,
        0.330,
    ]
    n1 = [
        0.466,
        0.466,
        0.466,
        0.457,
        0.369,
        0.326,
        0.328,
        0.316,
        0.310,
        0.317,
        0.305,
        0.301,
        0.300,
        0.304,
        0.303,
        0.292,
        0.296,
        0.289,
    ]
    sm1 = [
        0.534,
        0.534,
        0.534,
        0.536,
        0.463,
        0.414,
        0.411,
        0.401,
        0.396,
        0.378,
        0.389,
        0.389,
        0.381,
        0.369,
        0.391,
        0.385,
        0.384,
        0.381,
    ]

    steps = s0 + s1
    large = l0 + l1
    nano = n0 + n1
    small = sm0 + sm1

    ax.plot(steps, large, "-", color=C_LARGE, label="large (random init)", linewidth=2)
    ax.plot(steps, nano, "-", color=C_NANO, label="nano (warm-started)", linewidth=1.5)
    ax.plot(steps, small, "-", color=C_SMALL, label="small (warm-started)", linewidth=1.5)

    # Two key moments
    ax.axvline(x=2000, color="#aaa", linestyle=":", linewidth=0.8)
    ax.axvline(x=6550, color="#aaa", linestyle=":", linewidth=0.8)
    ax.text(2050, 0.92, "large\ncrashes", fontsize=7, color="#999")
    ax.text(6600, 0.92, "large\nre-emerges", fontsize=7, color=C_LARGE)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Expert Usage (EMA)")
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.62), framealpha=0.9)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, 12500)
    ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000, 12000])
    ax.grid(True)
    fig.savefig(OUT / "expert_usage_asym_warm.png")
    fig.savefig(OUT / "expert_usage_asym_warm.pdf")
    print("Saved expert_usage_asym_warm")
    plt.close()


if __name__ == "__main__":
    pareto_plot()
    waterfall_plot()
    expert_usage_plot()
    print("Done!")
