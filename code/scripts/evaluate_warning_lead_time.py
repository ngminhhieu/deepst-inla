#!/usr/bin/env python3
"""Measure how early a model first alerts for observed outbreak target-months.

For a given commune and target month, H3 is an earlier warning than H2/H1 because
its issue month is farther from the target. Among predictions that cross the alert
probability threshold, the largest horizon is therefore the first available alert.
This complements per-horizon precision/recall; it does not replace them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def warning_lead_metrics(predictions: pd.DataFrame) -> dict[str, float | int | str]:
    required = {
        "model",
        "fcode",
        "issue_date",
        "target_date",
        "horizon",
        "target_cases",
        "outbreak_threshold",
        "outbreak_probability",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing columns: {missing}")
    frame = predictions.copy()
    frame["issue_date"] = pd.to_datetime(frame["issue_date"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    if "outbreak_decision_threshold" in frame.columns:
        cutoff = pd.to_numeric(frame["outbreak_decision_threshold"], errors="raise")
    else:
        cutoff = pd.Series(0.5, index=frame.index)
    frame["observed_outbreak"] = frame["target_cases"] > frame["outbreak_threshold"]
    frame["alert"] = frame["outbreak_probability"] >= cutoff
    model_values = frame["model"].dropna().astype(str).unique()
    model = model_values[0] if len(model_values) == 1 else "mixed"

    # Collapse horizons for the same realised target month. The observed label
    # should be invariant across horizons; fail if inputs disagree.
    positive_leads: list[float] = []
    observed_positive = 0
    detected = 0
    for _, group in frame.groupby(["fcode", "target_date"], sort=False):
        labels = group["observed_outbreak"].unique()
        if len(labels) != 1:
            raise ValueError("Observed outbreak label differs across horizons for one target month")
        if not bool(labels[0]):
            continue
        observed_positive += 1
        alerted = group[group["alert"]]
        if alerted.empty:
            continue
        detected += 1
        earliest = alerted.loc[alerted["issue_date"].idxmin()]
        lead = (earliest["target_date"].year - earliest["issue_date"].year) * 12 + (
            earliest["target_date"].month - earliest["issue_date"].month
        )
        positive_leads.append(float(lead))

    leads = np.asarray(positive_leads, dtype=float)
    safe = lambda numerator, denominator: numerator / denominator if denominator else 0.0
    return {
        "model": model,
        "observed_outbreak_target_months": observed_positive,
        "detected_outbreak_target_months": detected,
        "target_month_detection_rate": safe(detected, observed_positive),
        "mean_warning_lead_months_detected": float(leads.mean()) if len(leads) else np.nan,
        "median_warning_lead_months_detected": float(np.median(leads)) if len(leads) else np.nan,
        "share_detected_with_lead_ge_2m": float(np.mean(leads >= 2)) if len(leads) else np.nan,
        "share_detected_with_lead_ge_3m": float(np.mean(leads >= 3)) if len(leads) else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.prediction_paths:
        if not path.exists():
            print(f"skip missing {path}")
            continue
        rows.append(warning_lead_metrics(pd.read_csv(path, dtype={"fcode": str})))
    if not rows:
        raise SystemExit("No prediction files found")
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
