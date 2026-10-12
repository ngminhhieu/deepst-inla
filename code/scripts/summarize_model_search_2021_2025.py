#!/usr/bin/env python3
"""Create the metrics-and-PNG-only delivery for the expanded model search."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def collect_test_metrics(new: Path, old: Path) -> pd.DataFrame:
    paths = {
        "INLA-TS": old / "INLA-ONLY-test/INLA-TS/metrics.csv",
        **{label: new / label / "metrics.csv" for label in (
            "TCN-GNN-BASE", "TCN-GNN-SOIL", "TCN-GNN-SOLAR",
            "TCN-GNN-DEWPOINT", "TCN-GNN-ET0", "TCN-GNN-ALL",
            "ARCH-LAST-GNN", "ARCH-TCN-GNN-NOGATE", "ARCH-TCN-GNN-GATED",
            "ARCH-LSTM-GNN", "ARCH-CNN-LSTM-GNN", "ARCH-WINNER-PEAK",
            "ARCH-TOP2-STACK",
        )},
        "ARCH-WINNER-REPR-INLA": new / "ARCH-WINNER-REPR-INLA/test/combined/metrics.csv",
        "PROPOSAL-PRECISION-GATE": new / "ARCH-WINNER-REPR-INLA-PEAK-GATE/metrics.csv",
        **{
            label: new / "ARCH-WINNER-INLA-ENSEMBLES" / label / "metrics.csv"
            for label in (
                "INLA-GNN-EXACT-SOFT-CRPS", "INLA-GNN-EXACT-SOFT-F1",
                "INLA-GNN-EXACT-GATE-CRPS", "INLA-GNN-EXACT-GATE-F1",
            )
        },
    }
    frames = []
    for label, path in paths.items():
        frame = pd.read_csv(path)
        frame["model"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["model", "horizon"])


def plot_ablation(feature: pd.DataFrame, architecture: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)
    panels = (
        (feature, "mean_crps", "Feature ablation: validation CRPS", False),
        (feature, "mean_f1", "Feature ablation: validation outbreak F1", True),
        (architecture, "mean_crps", "Architecture ablation: validation CRPS", False),
        (architecture, "mean_f1", "Architecture ablation: validation outbreak F1", True),
    )
    for axis, (frame, metric, title, higher) in zip(axes.flat, panels):
        ordered = frame.sort_values(metric, ascending=not higher)
        colors = ["#0072B2" if index == 0 else "#A8B0BA" for index in range(len(ordered))]
        axis.barh(ordered["model"], ordered[metric], color=colors)
        axis.invert_yaxis()
        axis.set_title(title, fontweight="bold")
        axis.grid(axis="x", alpha=0.2)
        for index, value in enumerate(ordered[metric]):
            axis.text(value, index, f" {value:.3f}", va="center", fontsize=8)
    fig.suptitle("Blocked validation 2019–2020; test 2021–2025 was not used for selection", fontsize=14)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=Path, default=ROOT / "results/feature_tcn_gnn_2021_2025_v1")
    parser.add_argument("--old", type=Path, default=ROOT / "results/eval_2021_2025_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/final_model_search_2021_2025")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    feature_metrics = pd.read_csv(args.new / "feature_ablation_validation_metrics.csv")
    feature_summary = pd.read_csv(args.new / "feature_ablation_validation_summary.csv")
    architecture_metrics = pd.read_csv(args.new / "architecture_validation_metrics.csv")
    architecture_summary = pd.read_csv(args.new / "architecture_validation_summary.csv")
    test_metrics = collect_test_metrics(args.new, args.old)

    feature_metrics.to_csv(args.output / "feature_ablation_validation_metrics.csv", index=False)
    architecture_metrics.to_csv(args.output / "architecture_ablation_validation_metrics.csv", index=False)
    test_metrics.to_csv(args.output / "all_models_test_metrics.csv", index=False)

    summary_columns = [
        "mae", "rmse", "crps", "outbreak_brier", "outbreak_precision",
        "outbreak_specificity", "outbreak_sensitivity", "outbreak_f1",
    ]
    summary = test_metrics.groupby("model", as_index=False)[summary_columns].mean()
    summary = summary.rename(columns={column: f"mean_H1_H3_{column}" for column in summary_columns})
    summary.sort_values("mean_H1_H3_crps").to_csv(
        args.output / "final_model_summary.csv", index=False
    )
    plot_ablation(feature_summary, architecture_summary, args.output / "Fig_ablation_summary.png")


if __name__ == "__main__":
    main()
