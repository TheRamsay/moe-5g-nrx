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


def _dense_run_filter(groups: list[str], run_role: str) -> str:
    quoted_groups = ", ".join(f"'{group}'" for group in groups)
    return (
        f"Config('experiment.batch_name') in [{quoted_groups}] "
        f"and Config('model.family') = 'static_dense' "
        f"and Config('registry.run_role') = '{run_role}'"
    )


def _panel_grid(runset: wr.Runset, panels: list[object]) -> wr.PanelGrid:
    return wr.PanelGrid(runsets=[runset], panels=panels)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    project = str(cfg.logging.get("project") or cfg.project.name)
    entity = cfg.logging.get("entity")
    if not entity:
        raise ValueError("logging.entity must be set to create a shared WandB report")

    train_groups = [str(group) for group in cfg.reporting.get("train_groups", [])]
    eval_groups = [str(group) for group in cfg.reporting.get("eval_groups", train_groups)]

    train_runs = wr.Runset(
        entity=str(entity),
        project=project,
        name="Dense training runs",
        filters=_dense_run_filter(train_groups, str(cfg.reporting.get("train_job_type", "train"))),
    )
    eval_runs = wr.Runset(
        entity=str(entity),
        project=project,
        name="Dense evaluation runs",
        filters=_dense_run_filter(eval_groups, str(cfg.reporting.get("eval_job_type", "evaluation"))),
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
                train_runs,
                [
                    wr.LinePlot(title="Training Loss", y=[wr.Metric("train/loss")]),
                    wr.LinePlot(title="Validation BER on UMa", y=[wr.Metric("val/uma/ber")]),
                    wr.LinePlot(title="Validation BER on TDL-C", y=[wr.Metric("val/tdlc/ber")]),
                ],
            ),
            wr.MarkdownBlock("## Final Evaluation"),
            _panel_grid(
                eval_runs,
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
                eval_runs,
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
            wr.MarkdownBlock("## Evaluation Comparison Table"),
            _panel_grid(
                eval_runs,
                [
                    wr.RunComparer(diff_only="split"),
                ],
            ),
            wr.MarkdownBlock(
                "## Lineage\n"
                "Use the evaluation run summaries to trace each point back to the source checkpoint artifact, "
                "the dataset artifacts used for `uma` / `tdlc`, and the local study folder recorded in "
                "`registry/study_path`."
            ),
        ],
        width="fluid",
    )

    report.save(draft=False)
    print(f"Report URL: {report.url}")


if __name__ == "__main__":
    main()
