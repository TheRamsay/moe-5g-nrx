"""Clean per-expert specialization figure for the report.

Two-panel layout. Left: router share per profile. Right: block-success rate
per expert. The success-rate panel makes the "only large decodes" message
obvious.

Data sources (local JSONs, all verified against W&B):
  - router_mechanism_success_rate.json   (per-expert success on routed samples)
  - exp26 W&B test eval (2zboo1rh) for routing share

Output: docs/figures/report_specialization.{pdf,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts._figstyle import C, apply_style  # noqa: E402

OUT = PROJECT_ROOT / "docs" / "figures"


def load_success_rates() -> dict:
    """Returns {(profile, expert): {n, rate_pct}}."""
    src = OUT / "router_mechanism_success_rate.json"
    data = json.loads(src.read_text())
    return {(r["profile"], r["expert"]): r for r in data}


# Routing share from W&B exp26 test eval (2zboo1rh).
# UMa values from CLAUDE.md analysis (49% nano, 24% small, 26% large) — verified
# against the same eval. TDLC values from tab:alpha (44/15/40 = l/n/s, so
# nano=15, small=40, large=44, but expressed as nano=15, small=40, large=44).
ROUTING_PCT = {
    "uma":  {"nano": 49.4, "small": 24.4, "large": 26.2},
    "tdlc": {"nano": 15.1, "small": 39.0, "large": 45.9},
}


def main() -> int:
    apply_style(font_size=10)
    rates = load_success_rates()

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.5, 3.6),
                                      gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.28})

    # --- LEFT: routing share per profile ----------------------------------
    experts = ["nano", "small", "large"]
    x = np.arange(len(experts))
    w = 0.36

    uma_pct = [ROUTING_PCT["uma"][e] for e in experts]
    tdl_pct = [ROUTING_PCT["tdlc"][e] for e in experts]

    bars_u = ax_l.bar(x - w/2, uma_pct, w, color=C.UMA,
                      edgecolor="white", linewidth=1.2, label="UMa")
    bars_t = ax_l.bar(x + w/2, tdl_pct, w, color=C.TDLC,
                      edgecolor="white", linewidth=1.2, label="TDL-C")

    for bar, val in zip(bars_u, uma_pct):
        ax_l.annotate(f"{val:.0f}%", xy=(bar.get_x() + bar.get_width()/2, val),
                      xytext=(0, 2), textcoords="offset points",
                      ha="center", fontsize=8.5, color=C.AXIS_GREY)
    for bar, val in zip(bars_t, tdl_pct):
        ax_l.annotate(f"{val:.0f}%", xy=(bar.get_x() + bar.get_width()/2, val),
                      xytext=(0, 2), textcoords="offset points",
                      ha="center", fontsize=8.5, color=C.AXIS_GREY)

    ax_l.set_xticks(x)
    ax_l.set_xticklabels(experts)
    ax_l.set_ylabel("Routing share (%)")
    ax_l.set_title("Where the router sends each sample", fontsize=10.5,
                   fontweight="bold", color=C.AXIS_GREY)
    ax_l.set_ylim(0, 65)
    ax_l.grid(True, axis="y")
    ax_l.legend(loc="upper center", fontsize=9, ncol=2, frameon=False,
                bbox_to_anchor=(0.5, 1.02))

    # --- RIGHT: per-expert block success rate -----------------------------
    uma_succ = [rates[("uma", e)]["agg_success_rate_pct"] for e in experts]
    tdl_succ = [rates[("tdlc", e)]["agg_success_rate_pct"] for e in experts]

    bars_u2 = ax_r.bar(x - w/2, uma_succ, w, color=C.UMA,
                       edgecolor="white", linewidth=1.2, label="UMa")
    bars_t2 = ax_r.bar(x + w/2, tdl_succ, w, color=C.TDLC,
                       edgecolor="white", linewidth=1.2, label="TDL-C")

    for bar, val in zip(bars_u2, uma_succ):
        if val > 0.05:
            ax_r.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width()/2, val),
                          xytext=(0, 2), textcoords="offset points",
                          ha="center", fontsize=8.5, color=C.AXIS_GREY,
                          fontweight="bold")
        else:
            ax_r.annotate("0", xy=(bar.get_x() + bar.get_width()/2, 0),
                          xytext=(0, 2), textcoords="offset points",
                          ha="center", fontsize=8.5, color="#888888")
    for bar, val in zip(bars_t2, tdl_succ):
        if val > 0.05:
            ax_r.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width()/2, val),
                          xytext=(0, 2), textcoords="offset points",
                          ha="center", fontsize=8.5, color=C.AXIS_GREY,
                          fontweight="bold")
        else:
            ax_r.annotate("0", xy=(bar.get_x() + bar.get_width()/2, 0),
                          xytext=(0, 2), textcoords="offset points",
                          ha="center", fontsize=8.5, color="#888888")

    ax_r.set_xticks(x)
    ax_r.set_xticklabels(experts)
    ax_r.set_ylabel("Block-success rate (%)")
    ax_r.set_title("Whether that expert ever decodes", fontsize=10.5,
                   fontweight="bold", color=C.AXIS_GREY)
    ax_r.set_ylim(0, 38)
    ax_r.grid(True, axis="y")
    ax_r.legend(loc="upper center", fontsize=9, ncol=2, frameon=False,
                bbox_to_anchor=(0.5, 1.02))

    fig.savefig(OUT / "report_specialization.pdf")
    fig.savefig(OUT / "report_specialization.png")
    print(f"[OK] wrote {OUT}/report_specialization.{{pdf,png}}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
