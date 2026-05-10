#!/usr/bin/env python
"""Regenerate the polished figures used by the final report."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._figstyle import C, apply_style  # noqa: E402

DOCS_FIG = PROJECT_ROOT / "docs" / "figures"
REPORT_FIG = PROJECT_ROOT / "report" / "obrazky-figures"
ENTITY = "knn_moe-5g-nrx"
PROJECT = "moe-5g-nrx"


def save_figure(fig: plt.Figure, stem: str, *, report_stem: str | None = None, png_dpi: int = 240) -> None:
    """Save a PDF/PNG pair under docs/figures and optionally sync it to the report."""

    DOCS_FIG.mkdir(parents=True, exist_ok=True)
    REPORT_FIG.mkdir(parents=True, exist_ok=True)

    pdf_path = DOCS_FIG / f"{stem}.pdf"
    png_path = DOCS_FIG / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=png_dpi)

    if report_stem is not None:
        shutil.copyfile(pdf_path, REPORT_FIG / f"{report_stem}.pdf")
        shutil.copyfile(png_path, REPORT_FIG / f"{report_stem}.png")

    print(f"[OK] wrote {pdf_path.relative_to(PROJECT_ROOT)} and .png")


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = "#333333",
    fontsize: int = 10,
    weight: str = "normal",
    radius: float = 0.025,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight)


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#444444",
    dashed: bool = False,
    rad: float = 0.0,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.35,
        linestyle=(0, (4, 3)) if dashed else "-",
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def plot_architecture() -> None:
    apply_style(font_size=9)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, 0.22, 0.84, 0.56, 0.09, "Received OFDM grid + LS estimate", facecolor="#F5F7FA", fontsize=11)
    add_box(ax, 0.22, 0.64, 0.56, 0.12, "Shared stem\n3-layer 2D CNN", facecolor="#D7CCFF", fontsize=12)
    add_box(ax, 0.33, 0.45, 0.34, 0.10, "Channel-aware router", facecolor="#FFE699", fontsize=11)
    add_box(ax, 0.08, 0.17, 0.22, 0.15, "nano\n4 blocks, dim 8\n320M FLOPs", facecolor="#CDEBFF", fontsize=9)
    add_box(ax, 0.39, 0.15, 0.22, 0.18, "small\n8 blocks, dim 32\n695M FLOPs", facecolor="#A9D8F5", fontsize=9)
    add_box(ax, 0.70, 0.12, 0.26, 0.24, "large\n8 blocks, dim 64\n1604M FLOPs", facecolor="#71BDF2", fontsize=10)
    add_box(ax, 0.22, 0.02, 0.56, 0.08, "LLR estimates", facecolor="#F5F7FA", fontsize=11)

    add_arrow(ax, (0.50, 0.84), (0.50, 0.76))
    add_arrow(ax, (0.50, 0.64), (0.50, 0.55))
    add_arrow(ax, (0.40, 0.45), (0.19, 0.32), color=C.HEADLINE, dashed=True, rad=0.04)
    add_arrow(ax, (0.50, 0.45), (0.50, 0.33), color=C.HEADLINE, dashed=True)
    add_arrow(ax, (0.60, 0.45), (0.83, 0.36), color=C.HEADLINE, dashed=True, rad=-0.06)
    add_arrow(ax, (0.19, 0.17), (0.42, 0.10))
    add_arrow(ax, (0.50, 0.15), (0.50, 0.10))
    add_arrow(ax, (0.83, 0.12), (0.61, 0.10))

    ax.text(0.81, 0.70, "285M FLOPs\nalways paid", color="#6F45D5", fontsize=9, fontweight="bold")
    ax.text(0.70, 0.47, "Gumbel-Softmax train\nhard top-1 inference", color=C.HEADLINE, fontsize=8.5)
    ax.text(0.03, 0.34, "one expert\nruns per slot", color="#666666", fontsize=8.5)

    save_figure(fig, "report_architecture")
    plt.close(fig)


def plot_pareto() -> None:
    apply_style(font_size=9)
    fig, ax = plt.subplots(figsize=(6.4, 4.1))

    dense_large_flops = 1_604.270464
    dense = [
        ("dense_nano", 320.208896 / dense_large_flops * 100, 0.951, C.DENSE_NANO),
        ("dense_small", 598.556672 / dense_large_flops * 100, 0.931, C.DENSE_SMALL),
        ("dense_large", 100.0, 0.901, C.DENSE_LARGE),
    ]
    moe_base = [
        ("alpha 5e-4", 100.0, 0.898),
        ("alpha 1e-3", 56.0, 0.906),
        ("selected MoE", 55.8, 0.902),
        ("alpha 5e-3", 60.0, 0.911),
    ]
    mode_b = [
        ("alpha 1e-3 + B", 46.7, 0.9066),
        ("selected MoE + B", 47.3, 0.9021),
        ("alpha 5e-3 + B", 41.9, 0.9116),
    ]

    ax.plot([x for _, x, _, _ in dense], [y for _, _, y, _ in dense], color=C.DENSE_NANO, alpha=0.45, lw=1.5)
    for label, x, y, color in dense:
        ax.scatter(x, y, s=70, marker="s", color=color, edgecolor="white", linewidth=0.9, zorder=5)
        offset = {"dense_nano": (7, -2), "dense_small": (7, -2), "dense_large": (-6, -4)}[label]
        ha = "right" if label == "dense_large" else "left"
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", ha=ha, fontsize=8.5, color=color)

    ax.scatter(
        [x for _, x, _ in moe_base],
        [y for _, _, y in moe_base],
        s=48,
        marker="o",
        facecolors="white",
        edgecolors=C.NEUTRAL_GREY,
        linewidth=1.3,
        zorder=4,
    )
    ax.scatter(55.8, 0.902, s=85, marker="o", color=C.NEUTRAL_GREY, edgecolor="white", linewidth=1.0, zorder=6)
    ax.annotate("selected MoE", (55.8, 0.902), xytext=(8, -20), textcoords="offset points", fontsize=8.5)

    ax.scatter(
        [x for label, x, _ in mode_b if label != "selected MoE + B"],
        [y for label, _, y in mode_b if label != "selected MoE + B"],
        s=52,
        marker="D",
        color=C.HEADLINE,
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
        alpha=0.85,
    )
    ax.scatter(47.3, 0.9021, s=260, marker="*", color=C.HEADLINE, edgecolor="white", linewidth=1.1, zorder=8)
    ax.annotate(
        "MoE + Mode B\n0.902 BLER, 47% FLOPs",
        (47.3, 0.9021),
        xytext=(-18, 26),
        textcoords="offset points",
        ha="right",
        fontsize=8.8,
        color=C.HEADLINE,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": C.HEADLINE, "lw": 0.9, "connectionstyle": "arc3,rad=-0.18"},
    )

    lmmse_bler = 0.8997
    ax.axhline(lmmse_bler, color=C.LMMSE, linestyle=":", linewidth=1.5)
    ax.text(
        76,
        lmmse_bler - 0.003,
        "LS-MRC",
        color=C.LMMSE,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )

    pareto = sorted([(dense[0][1], 0.951), (dense[1][1], 0.931), (47.3, 0.9021), (100.0, 0.901)])
    ax.plot([x for x, _ in pareto], [y for _, y in pareto], color=C.HEADLINE, alpha=0.18, lw=2.4, zorder=1)

    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=C.DENSE_SMALL, label="dense baselines"),
        Line2D(
            [0],
            [0],
            marker="o",
            color=C.NEUTRAL_GREY,
            markerfacecolor="white",
            label=r"trained MoE $\alpha$ sweep",
        ),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=C.HEADLINE, label="Mode B inference"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=8.2)
    ax.set_xlabel("Average realised FLOPs (% of dense_large)")
    ax.set_ylabel("Average BLER (lower is better)")
    ax.set_xlim(12, 110)
    ax.set_ylim(0.965, 0.892)
    ax.set_yticks([0.90, 0.92, 0.94, 0.96])
    ax.grid(True)

    save_figure(fig, "report_pareto", report_stem="report_pareto")
    plt.close(fig)


def load_success_rates() -> dict[tuple[str, str], dict[str, Any]]:
    data = json.loads((DOCS_FIG / "router_mechanism_success_rate.json").read_text())
    return {(row["profile"], row["expert"]): row for row in data}


def plot_specialization() -> None:
    apply_style(font_size=9)
    rates = load_success_rates()
    routing_pct = {
        "uma": {"nano": 49.4, "small": 24.4, "large": 26.2},
        "tdlc": {"nano": 15.1, "small": 39.0, "large": 45.9},
    }
    experts = ["nano", "small", "large"]
    x = np.arange(len(experts))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.28})
    for ax, panel in zip(axes, ["A", "B"]):
        ax.text(-0.10, 1.04, panel, transform=ax.transAxes, fontsize=11, fontweight="bold")

    uma = [routing_pct["uma"][expert] for expert in experts]
    tdlc = [routing_pct["tdlc"][expert] for expert in experts]
    axes[0].bar(x - width / 2, uma, width, color=C.UMA, edgecolor="white", linewidth=0.8, label="UMa")
    axes[0].bar(x + width / 2, tdlc, width, color=C.TDLC, edgecolor="white", linewidth=0.8, label="TDL-C")
    for xpos, values in [(x - width / 2, uma), (x + width / 2, tdlc)]:
        for xi, value in zip(xpos, values):
            axes[0].text(xi, value + 1.2, f"{value:.0f}%", ha="center", fontsize=8, color=C.AXIS_GREY)
    axes[0].set_title("Router allocation", fontsize=10)
    axes[0].set_ylabel("Routing share (%)")
    axes[0].set_ylim(0, 62)
    axes[0].set_xticks(x, experts)
    axes[0].grid(True, axis="y")

    uma_success = [rates[("uma", expert)]["agg_success_rate_pct"] for expert in experts]
    tdlc_success = [rates[("tdlc", expert)]["agg_success_rate_pct"] for expert in experts]
    axes[1].bar(x - width / 2, uma_success, width, color=C.UMA, edgecolor="white", linewidth=0.8, label="UMa")
    axes[1].bar(x + width / 2, tdlc_success, width, color=C.TDLC, edgecolor="white", linewidth=0.8, label="TDL-C")
    for xpos, values in [(x - width / 2, uma_success), (x + width / 2, tdlc_success)]:
        for xi, value in zip(xpos, values):
            label = f"{value:.1f}%" if value > 0 else "0%"
            axes[1].text(xi, value + 0.9, label, ha="center", fontsize=8, color=C.AXIS_GREY)
    axes[1].set_title("Block success after routing", fontsize=10)
    axes[1].set_ylabel("Block-success rate (%)")
    axes[1].set_ylim(0, 35)
    axes[1].set_xticks(x, experts)
    axes[1].grid(True, axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, "report_specialization", report_stem="router_mechanism_expert_specialization")
    plt.close(fig)


def fetch_history(run_id: str, keys: list[str], *, samples: int = 700) -> dict[str, np.ndarray]:
    import wandb

    run = wandb.Api().run(f"{ENTITY}/{PROJECT}/{run_id}")
    df = run.history(keys=keys, samples=samples, pandas=True).dropna(subset=["_step"])
    history = {"steps": df["_step"].to_numpy(dtype=np.float64)}
    for key in keys:
        history[key] = df[key].to_numpy(dtype=np.float64) if key in df.columns else np.full(len(df), np.nan)
    return history


def plot_routing_panel(ax: plt.Axes, history: dict[str, np.ndarray], title: str, panel: str) -> None:
    steps = history["steps"]
    nano = np.nan_to_num(history["train/ema/expert_usage/nano"])
    small = np.nan_to_num(history["train/ema/expert_usage/small"])
    large = np.nan_to_num(history["train/ema/expert_usage/large"])

    ax.stackplot(
        steps,
        nano,
        small,
        large,
        colors=[C.NANO, C.SMALL, C.LARGE],
        alpha=0.92,
        labels=["nano", "small", "large"],
    )
    ax.text(-0.08, 1.04, panel, transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 10_000)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True, axis="y")


def plot_trajectories(histories: dict[str, dict[str, np.ndarray]]) -> None:
    apply_style(font_size=9)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)

    plot_routing_panel(axes[0, 0], histories["phase1"], "Phase 1: cold start", "A")
    plot_routing_panel(axes[0, 1], histories["phase2"], "Phase 2: full warm-start", "B")
    plot_routing_panel(axes[1, 0], histories["asym"], "Asymmetric warm-start", "C")

    ax = axes[1, 1]
    ax.text(-0.08, 1.04, "D", transform=ax.transAxes, fontsize=11, fontweight="bold")
    entropy_specs = [
        ("Phase 1", histories["phase1"], C.NEUTRAL_GREY),
        ("Phase 2", histories["phase2"], C.HEADLINE),
        ("Asym warm", histories["asym"], C.LARGE),
    ]
    for label, history, color in entropy_specs:
        ax.plot(history["steps"], history["train/ema/router_entropy"], color=color, lw=1.7, label=label)
    ax.axhline(np.log(3), color="#888888", lw=0.9, ls="--", label=r"$\log 3$")
    ax.set_title("Router entropy", fontsize=10)
    ax.set_xlim(0, 10_000)
    ax.set_ylim(0, 1.2)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=7.8, frameon=True)

    axes[0, 0].set_ylabel("Routing share")
    axes[1, 0].set_ylabel("Routing share")
    axes[1, 0].set_xlabel("Training step")
    axes[1, 1].set_xlabel("Training step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()

    save_figure(fig, "report_trajectories", report_stem="routing_trajectories_collapse_modes")
    plt.close(fig)


def plot_wakeup(history: dict[str, np.ndarray]) -> None:
    apply_style(font_size=9)
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    steps = history["steps"]

    ax.plot(steps, history["train/ema/expert_usage/large"], color=C.LARGE, lw=2.2, label="large, random init")
    ax.plot(steps, history["train/ema/expert_usage/nano"], color=C.NANO, lw=1.8, label="nano, warm-started")
    ax.plot(steps, history["train/ema/expert_usage/small"], color=C.SMALL, lw=1.8, label="small, warm-started")
    ax.axvspan(0, 1500, color=C.LARGE, alpha=0.08)
    ax.axvspan(3000, 12_000, color=C.SMALL, alpha=0.08)
    ax.text(0.06, 0.92, "cold large transient", transform=ax.transAxes, color=C.LARGE, fontsize=8.3, fontweight="bold")
    ax.text(
        0.52,
        0.92,
        "stable heterogeneous routing",
        transform=ax.transAxes,
        color=C.SMALL,
        fontsize=8.3,
        fontweight="bold",
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Routing share")
    ax.set_xlim(0, 12_000)
    ax.set_ylim(-0.02, 1.08)
    ax.grid(True, axis="y")
    ax.legend(loc="center right", fontsize=8.0, frameon=True)

    save_figure(fig, "report_wakeup", report_stem="expert_usage_asym_warm")
    plt.close(fig)


def plot_pca_ood() -> None:
    apply_style(font_size=9)
    data = np.load(DOCS_FIG / "pca_ood_overlay.npz")
    uma = data["uma_proj"]
    tdlc = data["tdlc_proj"]
    asu = data["asu_proj"]
    in_dist = np.vstack([uma, tdlc])
    center = in_dist.mean(axis=0)
    spread = in_dist.std(axis=0)
    inside = (
        (asu[:, 0] >= center[0] - spread[0])
        & (asu[:, 0] <= center[0] + spread[0])
        & (asu[:, 1] >= center[1] - spread[1])
        & (asu[:, 1] <= center[1] + spread[1])
    )
    inside_pct = 100 * float(inside.mean())

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(uma[:, 0], uma[:, 1], s=8, color=C.UMA, alpha=0.22, label=f"UMa, n={len(uma)}", rasterized=True)
    ax.scatter(tdlc[:, 0], tdlc[:, 1], s=8, color=C.TDLC, alpha=0.22, label=f"TDL-C, n={len(tdlc)}", rasterized=True)
    ax.scatter(
        asu[:, 0],
        asu[:, 1],
        s=9,
        marker="x",
        linewidths=0.55,
        color=C.OOD,
        alpha=0.55,
        label=f"ASU ray-traced, n={len(asu)}",
        rasterized=True,
    )
    rect = Rectangle(
        (center[0] - spread[0], center[1] - spread[1]),
        2 * spread[0],
        2 * spread[1],
        fill=False,
        edgecolor="#555555",
        linewidth=1.2,
        linestyle="--",
        label=r"in-distribution $1\sigma$ box",
    )
    ax.add_patch(rect)
    ax.scatter(
        [uma[:, 0].mean(), tdlc[:, 0].mean(), asu[:, 0].mean()],
        [uma[:, 1].mean(), tdlc[:, 1].mean(), asu[:, 1].mean()],
        s=55,
        color=[C.UMA, C.TDLC, C.OOD],
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    ax.text(
        0.02,
        0.96,
        f"ASU inside 1-sigma box: {inside_pct:.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.95},
    )

    all_points = np.vstack([uma, tdlc, asu])
    xlim = np.percentile(all_points[:, 0], [0.2, 99.8])
    ylim = np.percentile(all_points[:, 1], [0.2, 99.8])
    ax.set_xlim(xlim[0] - 2, xlim[1] + 2)
    ax.set_ylim(ylim[0] - 2, ylim[1] + 2)
    ax.set_xlabel("PC 1, fitted on UMa + TDL-C")
    ax.set_ylabel("PC 2, fitted on UMa + TDL-C")
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=8.0, frameon=True)

    save_figure(fig, "pca_ood_overlay", report_stem="pca_ood_overlay")
    plt.close(fig)


def main() -> int:
    plot_architecture()
    plot_pareto()
    plot_specialization()
    keys = [
        "train/ema/expert_usage/nano",
        "train/ema/expert_usage/small",
        "train/ema/expert_usage/large",
        "train/ema/router_entropy",
    ]
    histories = {
        "phase1": fetch_history("2op33pak", keys),
        "phase2": fetch_history("89no8f1k", keys),
        "asym": fetch_history("t6lkdep2", keys),
    }
    plot_trajectories(histories)
    plot_wakeup(histories["asym"])
    plot_pca_ood()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
