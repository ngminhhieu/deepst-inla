#!/usr/bin/env python3
"""Create compact metrics deliverables for the fixed-model split rerun."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SOURCES = {
    "INLA-TS": "INLA-ONLY-test/INLA-TS/metrics.csv",
    "TCN-GNN Base": "TCN-GNN-BASE/metrics.csv",
    "DeepST-Alert (proposed deep)": "DEEPST-ALERT/metrics.csv",
    "TCN-GNN Deep Ensemble (variant)": "TCN-GNN-DEEP-ENSEMBLE/metrics.csv",
    "Representation-INLA (variant)": "REPRESENTATION-INLA/test/combined/metrics.csv",
    "DeepST-INLA (proposed hybrid)": "DEEPST-INLA/metrics.csv",
}
LOSS_COLUMNS = ("mae", "rmse", "crps", "outbreak_brier")
GAIN_COLUMNS = (
    "outbreak_precision",
    "outbreak_specificity",
    "outbreak_sensitivity",
    "outbreak_f1",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frames = []
    for label, relative in SOURCES.items():
        frame = pd.read_csv(args.run_root / relative)
        frame["model"] = label
        frames.append(frame)
    by_horizon = pd.concat(frames, ignore_index=True).sort_values(["model", "horizon"])
    metrics = list(LOSS_COLUMNS + GAIN_COLUMNS)
    summary = by_horizon.groupby("model", as_index=False)[metrics].mean()
    for column in metrics:
        summary[f"rank_{column}"] = summary[column].rank(
            ascending=column in LOSS_COLUMNS, method="min"
        )
    rank_columns = [f"rank_{column}" for column in metrics]
    summary["mean_metric_rank"] = summary[rank_columns].mean(axis=1)
    summary["worst_metric_rank"] = summary[rank_columns].max(axis=1)
    summary = summary.sort_values(["mean_metric_rank", "worst_metric_rank"])

    by_horizon.to_csv(args.output / "final_metrics_by_horizon.csv", index=False)
    summary.to_csv(args.output / "final_metrics_mean_H1_H3.csv", index=False)
if __name__ == "__main__":
    main()
