#!/usr/bin/env python
"""Build the report SNR waterfall figure from MetaCentrum run logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

NEURAL_PATTERN = r"^\s*([+-]?\d+\.\d+)\s*-\s*([+-]?\d+\.\d+)" r"\s*\|\s*([0-9.]+)\s*\|\s*([0-9,]+)"  # noqa: E501
LS_PATTERN = r"^\s*\[\s*([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]\s*dB\s*" r"\(n=\s*([0-9,]+)\):\s*BLER=([0-9.]+)"  # noqa: E501
NEURAL_ROW = re.compile(NEURAL_PATTERN)
LS_ROW = re.compile(LS_PATTERN)


def _empty_profile() -> dict[str, list[float] | list[int]]:
    return {"left": [], "right": [], "center": [], "bler": [], "samples": []}


def _append_row(
    target: dict[str, list[float] | list[int]], left: float, right: float, bler: float, samples: int
) -> None:
    target["left"].append(left)
    target["right"].append(right)
    target["center"].append((left + right) / 2.0)
    target["bler"].append(bler)
    target["samples"].append(samples)


def parse_neural_log(path: Path) -> dict[str, dict[str, list[float] | list[int]]]:
    out = {"uma": _empty_profile(), "tdlc": _empty_profile()}
    current_profile: str | None = None
    in_table = False
    for line in path.read_text().splitlines():
        profile_match = re.search(r"Results for:\s*(uma|tdlc)", line)
        if profile_match:
            current_profile = profile_match.group(1)
            in_table = False
            continue
        if "SNR-Binned BLER" in line and current_profile is not None:
            in_table = True
            continue
        if not in_table or current_profile is None:
            continue
        row = NEURAL_ROW.match(line)
        if row:
            left, right, bler, samples = row.groups()
            _append_row(out[current_profile], float(left), float(right), float(bler), int(samples.replace(",", "")))
        elif out[current_profile]["bler"] and not line.strip():
            in_table = False
    return out


def parse_ls_log(path: Path) -> dict[str, dict[str, list[float] | list[int]]]:
    out = {"uma": _empty_profile(), "tdlc": _empty_profile()}
    current_profile: str | None = None
    in_table = False
    for line in path.read_text().splitlines():
        profile_match = re.search(r"===\s*(uma|tdlc)\s*===", line)
        if profile_match:
            current_profile = profile_match.group(1)
            in_table = False
            continue
        if "Per-SNR BLER" in line and current_profile is not None:
            in_table = True
            continue
        if not in_table or current_profile is None:
            continue
        row = LS_ROW.match(line)
        if row:
            left, right, samples, bler = row.groups()
            _append_row(out[current_profile], float(left), float(right), float(bler), int(samples.replace(",", "")))
        elif out[current_profile]["bler"] and not line.strip():
            in_table = False
    return out


def build_source(exp26_log: Path, dense_log: Path, ls_log: Path) -> dict:
    exp26 = parse_neural_log(exp26_log)
    dense = parse_neural_log(dense_log)
    ls_mrc = parse_ls_log(ls_log)
    return {
        "source": {
            "exp26": "MetaCentrum job 19583604, W&B run rjbc09a4",
            "dense_large": "MetaCentrum job 19583605, W&B run m2tlouu3",
            "ls_mrc": "MetaCentrum job 19583606, lmmse_snr20_results.json",
            "test_split": "dense-v1/test, 32768 samples per profile",
        },
        "profiles": {
            profile: {
                "exp26": exp26[profile],
                "dense_large": dense[profile],
                "ls_mrc": ls_mrc[profile],
            }
            for profile in ("uma", "tdlc")
        },
    }


def plot(source: dict, output_pdf: Path, output_png: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 180,
            "savefig.dpi": 320,
        }
    )
    styles = {
        "ls_mrc": {"label": "LS-MRC", "color": "#117733", "marker": "o", "linewidth": 1.9},
        "dense_large": {"label": "dense large", "color": "#444444", "marker": "s", "linewidth": 1.7},
        "exp26": {"label": "selected MoE", "color": "#0072B2", "marker": "D", "linewidth": 1.7},
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85), sharey=True)
    for ax, profile, title in zip(axes, ("tdlc", "uma"), ("TDL-C", "UMa")):
        for key in ("ls_mrc", "dense_large", "exp26"):
            series = source["profiles"][profile][key]
            ax.plot(series["center"], series["bler"], markersize=3.1, **styles[key])
        ax.set_title(title)
        ax.set_xlabel("SNR [dB]")
        ax.grid(True, which="major", alpha=0.28, linewidth=0.6)
        ax.grid(True, which="minor", alpha=0.12, linewidth=0.4)
        ax.set_yscale("log")
        ax.set_ylim(0.04, 1.08)
        ax.set_yticks([1.0, 0.5, 0.2, 0.1, 0.05])
        ax.set_yticklabels(["1.0", "0.5", "0.2", "0.1", "0.05"])
    axes[0].set_ylabel("BLER")
    axes[0].legend(loc="lower left", frameon=False)
    fig.tight_layout(w_pad=1.2)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp26-log", type=Path, required=True)
    parser.add_argument("--dense-log", type=Path, required=True)
    parser.add_argument("--ls-log", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, default=Path("docs/figures/snr_waterfall_20bin_source.json"))
    parser.add_argument("--output-pdf", type=Path, default=Path("report/obrazky-figures/snr_waterfall_report.pdf"))
    parser.add_argument("--output-png", type=Path, default=Path("report/obrazky-figures/snr_waterfall_report.png"))
    args = parser.parse_args()

    source = build_source(args.exp26_log, args.dense_log, args.ls_log)
    args.source_json.parent.mkdir(parents=True, exist_ok=True)
    args.source_json.write_text(json.dumps(source, indent=2) + "\n")
    plot(source, args.output_pdf, args.output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
