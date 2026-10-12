#!/usr/bin/env python3
"""Evaluate resource-constrained outbreak capture from row-level predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--budgets", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.30]
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/pending/decision_utility.csv")
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/pending/decision_utility_summary.csv"),
    )
    args = parser.parse_args()
    if any(value <= 0 or value > 1 for value in args.budgets):
        raise ValueError("Budgets must be proportions in (0, 1]")

    data = pd.read_csv(args.predictions)
    required = {
        "model",
        "horizon",
        "target_date",
        "target_cases",
        "outbreak_threshold",
        "outbreak_probability",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")
    data["target_date"] = pd.to_datetime(data["target_date"])
    data["observed_outbreak"] = data.target_cases > data.outbreak_threshold

    rows = []
    for (model, horizon, date), frame in data.groupby(
        ["model", "horizon", "target_date"]
    ):
        frame = frame.sort_values("outbreak_probability", ascending=False)
        positives = int(frame.observed_outbreak.sum())
        for budget in args.budgets:
            selected = max(1, int(np.ceil(len(frame) * budget)))
            captured = int(frame.head(selected).observed_outbreak.sum())
            rows.append(
                {
                    "model": model,
                    "horizon": int(horizon),
                    "target_date": date,
                    "budget": budget,
                    "selected": selected,
                    "outbreaks": positives,
                    "captured": captured,
                    "capture_rate": captured / positives if positives else np.nan,
                    "selection_precision": captured / selected,
                }
            )
    output = pd.DataFrame(rows)

    # Pool counts across target months so months with many outbreaks contribute
    # in proportion to the number of response opportunities. The monthly mean is
    # retained separately as a stability diagnostic.
    summary = (
        output.groupby(["model", "horizon", "budget"], as_index=False)
        .agg(
            target_months=("target_date", "nunique"),
            selected=("selected", "sum"),
            outbreaks=("outbreaks", "sum"),
            captured=("captured", "sum"),
            monthly_capture_rate_mean=("capture_rate", "mean"),
            monthly_capture_rate_sd=("capture_rate", "std"),
        )
    )
    summary["pooled_capture_rate"] = summary.captured / summary.outbreaks
    summary["selection_precision"] = summary.captured / summary.selected
    summary["random_capture_expectation"] = summary.budget
    summary["capture_lift_over_random"] = (
        summary.pooled_capture_rate / summary.random_capture_expectation
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    summary.to_csv(args.summary, index=False)
    provenance = {
        "predictions": f"retained_model_output/{args.predictions.name}",
        "predictions_sha256": sha256(args.predictions),
        "ranking": "descending outbreak_probability within target month and horizon",
        "budget_rounding": "ceil(number_of_communes * budget)",
        "budgets": args.budgets,
    }
    args.summary.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
