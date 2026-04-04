#!/usr/bin/env python

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wandb_workspaces.reports import v2 as wr  # noqa: E402


def _dense_run_filter(cfg: DictConfig) -> str:
    groups = [str(group) for group in cfg.reporting.get("train_groups", [])]
    quoted_groups = ", ".join(f"'{group}'" for group in groups)
    return f"Config('experiment.batch_name') in [{quoted_groups}] and Config('model.family') = 'static_dense'"


def _panel_grid(runset: wr.Runset, panels: list[object]) -> wr.PanelGrid:
    return wr.PanelGrid(runsets=[runset], panels=panels)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    project = str(cfg.logging.get("project") or cfg.project.name)
    entity = cfg.logging.get("entity")
    if not entity:
        raise ValueError("logging.entity must be set to create a shared WandB report")

    dense_runs = wr.Runset(
        entity=str(entity),
        project=project,
        name="Dense study runs",
        filters=_dense_run_filter(cfg),
    )

    report = wr.Report(
        project=project,
        entity=str(entity),
        title=str(cfg.reporting.title),
        description=str(cfg.reporting.description),
        blocks=[
            wr.MarkdownBlock(
                "# Dense Baseline Study\n"
                "This report compares dense baseline and capacity runs. Training runs contribute time-series curves, "
                "and evaluation runs contribute final test-point metrics linked to checkpoint and dataset artifacts."
            ),
            wr.TableOfContents(),
            wr.MarkdownBlock("## Training Curves"),
            _panel_grid(
                dense_runs,
                [
                    wr.LinePlot(title="Training Loss", y=[wr.Metric("train/loss")]),
                    wr.LinePlot(title="Validation BLER on UMa", y=[wr.Metric("val/uma/bler")]),
                    wr.LinePlot(title="Validation BLER on TDL-C", y=[wr.Metric("val/tdlc/bler")]),
                ],
            ),
            wr.MarkdownBlock("## Final Evaluation"),
            _panel_grid(
                dense_runs,
                [
                    wr.BarPlot(
                        title="Test BLER by Profile",
                        metrics=[wr.SummaryMetric("eval/uma/bler"), wr.SummaryMetric("eval/tdlc/bler")],
                        orientation="v",
                    ),
                    wr.BarPlot(
                        title="Test BER by Profile",
                        metrics=[wr.SummaryMetric("eval/uma/ber"), wr.SummaryMetric("eval/tdlc/ber")],
                        orientation="v",
                    ),
                ],
            ),
            wr.MarkdownBlock("## Capacity Tradeoffs"),
            _panel_grid(
                dense_runs,
                [
                    wr.ScatterPlot(
                        title="Parameters vs TDL-C BLER",
                        x=wr.SummaryMetric("model/num_parameters"),
                        y=wr.SummaryMetric("eval/tdlc/bler"),
                    ),
                    wr.ScatterPlot(
                        title="Parameters vs UMa BLER",
                        x=wr.SummaryMetric("model/num_parameters"),
                        y=wr.SummaryMetric("eval/uma/bler"),
                    ),
                ],
            ),
            wr.MarkdownBlock("## Run Comparison"),
            _panel_grid(
                dense_runs,
                [
                    wr.RunComparer(diff_only="split"),
                ],
            ),
        ],
        width="fluid",
    )

    report.save(draft=False)
    print(f"Report URL: {report.url}")


if __name__ == "__main__":
    main()
