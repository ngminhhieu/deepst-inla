#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dengue_h3.evaluation import outbreak_classification_metrics


ORDER = [
    "seasonal_naive",
    "province_allocation",
    "INLA-T",
    "INLA-S",
    "INLA-TS",
    "CNN",
    "LSTM",
    "CNN-LSTM",
    "GNN",
    "LSTM-FULL",
    "LSTM-GNN",
    "CNN-LSTM-FULL",
    "CNN-LSTM-GNN",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M6-INLA",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results")
    args = parser.parse_args()
    root = Path(args.results_dir).resolve()
    frames = []
    for model in ORDER:
        path = root / model / "metrics.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame = _add_outbreak_classification(root / model, frame)
            frame.to_csv(path, index=False)
            frames.append(frame)
    if not frames:
        raise SystemExit("No metrics.csv files were found")
    summary = pd.concat(frames, ignore_index=True)
    summary["model"] = pd.Categorical(summary["model"], ORDER, ordered=True)
    summary = summary.sort_values("model")
    compare = root / "compare"
    compare.mkdir(parents=True, exist_ok=True)
    summary.to_csv(compare / "all_horizon_metrics.csv", index=False)
    summary[summary["horizon"] == 3].to_csv(
        compare / "primary_h3_metrics.csv", index=False
    )


def _add_outbreak_classification(model_dir: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    predictions_path = model_dir / "predictions.csv"
    if not predictions_path.exists():
        raise SystemExit(f"Missing predictions required for classification: {predictions_path}")
    predictions = pd.read_csv(predictions_path)
    output = metrics.copy()
    for index, row in output.iterrows():
        frame = predictions[predictions["horizon"] == row["horizon"]]
        classification = outbreak_classification_metrics(
            frame["target_cases"].to_numpy(float),
            frame["outbreak_threshold"].to_numpy(float),
            frame["outbreak_probability"].to_numpy(float),
        )
        for name, value in classification.items():
            output.loc[index, name] = value
    return output


if __name__ == "__main__":
    main()
