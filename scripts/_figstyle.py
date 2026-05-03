"""Single source of truth for figure style across the report.

Wong color-blind safe palette from Nature Methods 2011, with consistent
typography, spacing, and grid rules. Import via:

    from scripts._figstyle import apply_style, C
    apply_style()
    plt.scatter(x, y, color=C.NANO)
"""

from __future__ import annotations

import matplotlib


class C:
    """Wong color-blind safe palette."""

    NANO = "#56B4E9"   # sky blue (cool, low-cost)
    SMALL = "#009E73"  # bluish green (mid)
    LARGE = "#0072B2"  # blue (high-cost, anchor)
    DARK_BLUE = "#003D60"  # darker variant of LARGE for emphasis

    DENSE_NANO = "#56B4E9"
    DENSE_SMALL = "#0072B2"
    DENSE_LARGE = "#003D60"

    HEADLINE = "#D55E00"   # vermilion (the headline result, accent)
    LMMSE = "#117733"      # dark green (classical)

    NEUTRAL_GREY = "#999999"
    AXIS_GREY = "#444444"

    UMA = "#56B4E9"
    TDLC = "#0072B2"
    OOD = "#D55E00"


def apply_style(font_size: int = 10) -> None:
    """Apply the standard rcParams. Call once before plotting."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": font_size,
            "axes.titlesize": font_size + 1,
            "axes.titleweight": "bold",
            "axes.labelsize": font_size + 1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": C.AXIS_GREY,
            "axes.labelcolor": C.AXIS_GREY,
            "xtick.color": C.AXIS_GREY,
            "ytick.color": C.AXIS_GREY,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#cccccc",
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.5,
            "grid.color": "#bbbbbb",
            "axes.axisbelow": True,
        }
    )
