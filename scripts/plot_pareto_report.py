"""Clean Pareto BLER vs FLOPs figure for the report.

Single-column friendly. All data verified against W&B summaries and analysis JSONs.
Output: docs/figures/report_pareto.{pdf,png}

Verified points:
  Dense neural baselines (W&B `bx7hylp6`/`8haq7zuz`/`gpmfhn6k`):
    nano  : avg BLER 0.951, FLOPs   320 M ( 20.0% of dense_large)
    small : avg BLER 0.931, FLOPs   695 M ( 43.3%)
    large : avg BLER 0.901, FLOPs  1604 M (100.0%)

  Trained MoE α-sweep (eval IDs `002cwsy2`/`5jswm490`/`2zboo1rh`/`dh4x0qmu`):
    exp24 (α=5e-4): avg BLER 0.898, FLOPs 100.0%   (router collapsed to large)
    exp25 (α=1e-3): avg BLER 0.906, FLOPs  56.0%
    exp26 (α=2e-3): avg BLER 0.902, FLOPs  55.8%   (training-time Pareto knee)
    exp27 (α=5e-3): avg BLER 0.911, FLOPs  60.0%

  Mode B inference (zero-out small at deploy; from `inference_mask_exp{25,26,27}.json`):
    exp25 + B : avg BLER 0.907, FLOPs  46.7%
    exp26 + B : avg BLER 0.902, FLOPs  47.3%   <-- HEADLINE
    exp27 + B : avg BLER 0.912, FLOPs  41.9%

  Classical LMMSE LS-MRC (job 19583606 lmmse_snr20_results.json):
    avg BLER 0.900, no neural FLOPs
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "docs" / "figures"

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 9,
    "legend.framealpha": 0.95,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})

# Colour scheme — Wong colour-blind safe (matches the poster).
C_DENSE = "#0072B2"        # blue
C_DENSE_LIGHT = "#56B4E9"  # sky blue (for nano)
C_DENSE_DARK = "#003D60"   # dark blue (for large)
C_TRAINED = "#999999"      # neutral grey for α-sweep
C_HEADLINE = "#D55E00"     # vermilion — Mode B headline
C_MODE_B = "#D55E00"
C_LMMSE = "#009E73"        # bluish green — classical reference

fig, ax = plt.subplots(figsize=(6.0, 4.2))

# Dense ladder: connecting line + labelled points
dense = [
    ("dense_nano",  20.0, 0.951, C_DENSE_LIGHT),
    ("dense_small", 43.3, 0.931, C_DENSE),
    ("dense_large", 100.0, 0.901, C_DENSE_DARK),
]
ax.plot([p[1] for p in dense], [p[2] for p in dense],
        "-", color=C_DENSE, alpha=0.35, linewidth=1.5, zorder=2)
for label, x, y, c in dense:
    ax.scatter(x, y, marker="s", s=90, color=c, zorder=6,
               edgecolors="white", linewidths=1.0)
    if "nano" in label:
        ax.annotate(label, (x, y), xytext=(8, -2),
                    textcoords="offset points", fontsize=9, color=c)
    elif "small" in label:
        ax.annotate(label, (x, y), xytext=(8, -2),
                    textcoords="offset points", fontsize=9, color=c)
    else:
        ax.annotate(label, (x, y), xytext=(-8, -3),
                    textcoords="offset points", fontsize=9, color=c, ha="right")

# α-sweep trained MoE (baseline inference)
alpha_sweep = [
    (100.0, 0.898),
    ( 56.0, 0.906),
    ( 55.8, 0.902),
    ( 60.0, 0.911),
]
ax.scatter([p[0] for p in alpha_sweep], [p[1] for p in alpha_sweep],
           marker="o", s=55, facecolors="white", edgecolors=C_TRAINED,
           linewidths=1.4, zorder=5, label=r"Trained MoE ($\alpha$ sweep, baseline inference)")

# exp26 baseline: filled, slightly bigger
ax.scatter(55.8, 0.902, marker="o", s=110, color=C_TRAINED,
           edgecolors="white", linewidths=1.5, zorder=7)
ax.annotate("exp26 (trained MoE)", (55.8, 0.902), xytext=(10, 6),
            textcoords="offset points", fontsize=9, color="#444")

# Mode B sweep
mode_b = [
    (46.7, 0.907),
    (47.3, 0.902),
    (41.9, 0.912),
]
ax.scatter([p[0] for p in mode_b[:1] + mode_b[2:]],
           [p[1] for p in mode_b[:1] + mode_b[2:]],
           marker="D", s=55, color=C_MODE_B, edgecolors="white",
           linewidths=1.0, zorder=7, alpha=0.85,
           label="Trained MoE + Mode B inference")

# Mode B headline (exp26 + B)
ax.scatter(47.3, 0.902, marker="*", s=380, color=C_HEADLINE,
           edgecolors="white", linewidths=1.5, zorder=10,
           label="exp26 + Mode B (ours, no retraining)")
ax.annotate("exp26 + Mode B", (47.3, 0.902), xytext=(-14, 18),
            textcoords="offset points", fontsize=10, color=C_HEADLINE,
            ha="right", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_HEADLINE, lw=1.0,
                            connectionstyle="arc3,rad=-0.2"))

# LMMSE classical reference (horizontal line)
ax.axhline(0.900, color=C_LMMSE, linestyle=":", linewidth=1.5, zorder=2,
           label="LMMSE LS-MRC (classical, no neural compute)")

# Pareto frontier through the four dominating neural points
pareto_pts = sorted([(20.0, 0.951), (43.3, 0.931), (47.3, 0.902), (100.0, 0.901)])
ax.plot([p[0] for p in pareto_pts], [p[1] for p in pareto_pts],
        "-", color=C_HEADLINE, alpha=0.18, linewidth=2.5, zorder=1)

ax.set_xlabel("Average realised FLOPs (% of dense_large)")
ax.set_ylabel("Average BLER (lower is better)")
ax.set_xlim(10, 110)
ax.set_ylim(0.890, 0.965)
ax.invert_yaxis()
ax.grid(True)
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1,
                frameon=False, fontsize=8.5)

fig.savefig(OUT / "report_pareto.pdf")
fig.savefig(OUT / "report_pareto.png", dpi=200)
print(f"[OK] wrote {OUT}/report_pareto.pdf and .png")
