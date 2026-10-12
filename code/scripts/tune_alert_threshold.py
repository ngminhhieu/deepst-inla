#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_h3.evaluation import outbreak_classification_metrics, score_predictions


def select_thresholds(
    validation: pd.DataFrame,
    *,
    minimum_precision: float,
) -> tuple[dict[int, float], dict[str, list[dict[str, float]]]]:
    thresholds: dict[int, float] = {}
    grids: dict[str, list[dict[str, float]]] = {}
    candidates = np.round(np.arange(0.05, 0.81, 0.01), 2)
    for horizon in sorted(validation["horizon"].unique()):
        part = validation[validation["horizon"].eq(horizon)]
        scores = []
        for threshold in candidates:
            metrics = outbreak_classification_metrics(
                part["target_cases"].to_numpy(float),
                part["outbreak_threshold"].to_numpy(float),
                part["outbreak_probability"].to_numpy(float),
                decision_threshold=float(threshold),
            )
            scores.append(
                {
                    "threshold": float(threshold),
                    "f1": float(metrics["outbreak_f1"]),
                    "precision": float(metrics["outbreak_precision"]),
                    "recall": float(metrics["outbreak_recall"]),
                    "specificity": float(metrics["outbreak_specificity"]),
                }
            )
        eligible = [
            row for row in scores if row["precision"] >= minimum_precision
        ]
        pool = eligible or scores
        best = max(
            pool,
            key=lambda row: (
                row["f1"],
                row["recall"],
                row["specificity"],
                -abs(row["threshold"] - 0.5),
            ),
        )
        thresholds[int(horizon)] = float(best["threshold"])
        grids[str(int(horizon))] = scores
    return thresholds, grids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--minimum-precision", type=float, default=0.50)
    args = parser.parse_args()

    validation = pd.read_csv(
        args.result_dir / "validation_predictions.csv", dtype={"fcode": str}
    )
    predictions = pd.read_csv(
        args.result_dir / "predictions.csv", dtype={"fcode": str}
    )
    samples = np.load(args.result_dir / "predictive_samples.npz")["samples"]
    thresholds, grids = select_thresholds(
        validation,
        minimum_precision=args.minimum_precision,
    )
    output = predictions.copy()
    output["model"] = args.model_label
    output["outbreak_decision_threshold"] = output["horizon"].map(thresholds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "predictions.csv", index=False)
    np.savez_compressed(args.output_dir / "predictive_samples.npz", samples=samples)
    score_predictions(
        output,
        samples,
        decision_threshold=thresholds,
    ).to_csv(args.output_dir / "metrics.csv", index=False)
    (args.output_dir / "threshold_selection.json").write_text(
        json.dumps(
            {
                "model": args.model_label,
                "selection_split": "validation",
                "minimum_precision": args.minimum_precision,
                "thresholds_by_horizon": thresholds,
                "validation_grid": grids,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
