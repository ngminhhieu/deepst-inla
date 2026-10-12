#!/usr/bin/env python3
"""Build an RHC-level comparison under one harmonized MOH ground truth."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from build_final_commune_rhc_alerts import read_alert_workbook


def rhc_name(value: str) -> str:
    parts = re.sub(r"^TTYTKV_", "", value).split("_")
    if len(parts) > 1 and len(parts[-1]) <= 3:
        parts = parts[:-1]
    return "TTYT khu vực " + " ".join(parts).title()


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    labels, predictions = labels.astype(bool), predictions.astype(bool)
    tp = int(np.sum(labels & predictions)); fp = int(np.sum(~labels & predictions))
    fn = int(np.sum(labels & ~predictions)); tn = int(np.sum(~labels & ~predictions))
    safe = lambda a, b: float(a / b) if b else 0.0
    precision = safe(tp, tp + fp); sensitivity = safe(tp, tp + fn)
    specificity = safe(tn, tn + fp)
    return {
        "n": len(labels), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": safe(tp + tn, len(labels)), "precision": precision,
        "sensitivity": sensitivity, "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "f1": safe(2 * precision * sensitivity, precision + sensitivity),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--province-lookup", type=Path, required=True)
    parser.add_argument("--current-alerts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions, dtype={"fcode": str})
    frame["target_date"] = pd.to_datetime(frame.target_date).dt.to_period("M").dt.to_timestamp()
    thresholds = pd.read_csv(args.thresholds, dtype={"fcode": str})
    thresholds["target_date"] = pd.to_datetime(dict(year=thresholds.year, month=thresholds.month, day=1))
    frame = frame.merge(
        thresholds[["fcode", "target_date", "alert_threshold_moh"]],
        on=["fcode", "target_date"], how="left", validate="one_to_one",
    )
    observations = pd.read_csv(
        args.observations, usecols=["fcode", "year", "month", "obs_dengue_cases", "population"],
        dtype={"fcode": str},
    )
    observations["target_date"] = pd.to_datetime(dict(
        year=observations.year, month=observations.month, day=1
    ))
    frame = frame.merge(
        observations[["fcode", "target_date", "obs_dengue_cases", "population"]],
        on=["fcode", "target_date"], how="left", suffixes=("", "_observed"),
        validate="one_to_one",
    )
    province = pd.read_csv(args.province_lookup, dtype={"fcode": str, "province_code": str})
    frame = frame.merge(province, on="fcode", how="left", validate="many_to_one")

    current = read_alert_workbook(args.current_alerts)
    current_keys = set(zip(current.target_date, current.rhc_id))
    frame["current_rhc_alert"] = pd.array([
        int((row.target_date, row.fcode) in current_keys) for row in frame.itertuples()
    ], dtype="Int64")
    frame["proposed_rhc_alert"] = pd.array(
        frame.outbreak_probability >= frame.outbreak_decision_threshold, dtype="Int64"
    )
    frame["distribution_moh_alert"] = pd.array(
        frame.distribution_moh_probability >= 0.50, dtype="Int64"
    )
    frame["moh_groundtruth"] = pd.array([
        pd.NA if pd.isna(cases) else int(float(cases) > float(threshold))
        for cases, threshold in zip(frame.obs_dengue_cases, frame.alert_threshold_moh)
    ], dtype="Int64")
    frame["forecast_month"] = frame.target_date.dt.strftime("%Y-%m")
    frame["rhc_name"] = frame.fcode.map(rhc_name)
    frame = frame.rename(columns={
        "fcode": "rhc_id", "predicted_mean": "rhc_prediction",
        "obs_dengue_cases": "observed_cases", "alert_threshold_moh": "moh_threshold_rhc",
        "outbreak_probability": "proposed_alert_probability",
    })
    columns = [
        "forecast_month", "rhc_id", "rhc_name", "province_code", "province",
        "rhc_prediction", "proposed_alert_probability", "proposed_rhc_alert",
        "distribution_moh_probability", "distribution_moh_alert", "current_rhc_alert",
        "observed_cases", "moh_threshold_rhc", "moh_groundtruth",
    ]
    output = frame[columns].sort_values(["forecast_month", "rhc_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, na_rep="NA")

    available = output.loc[output.moh_groundtruth.notna()].copy()
    metrics = []
    for model, column in [
        ("Proposed hybrid RHC alert", "proposed_rhc_alert"),
        ("Hybrid distribution P(MOH)>0.5", "distribution_moh_alert"),
        ("Current RHC alert harmonized to MOH GT", "current_rhc_alert"),
    ]:
        metrics.append({"model": model, **binary_metrics(
            available.moh_groundtruth.to_numpy(int), available[column].to_numpy(int)
        )})
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(args.output.parent / "comparison_metrics.csv", index=False)

    monthly_rows = []
    for month, part in output.groupby("forecast_month", sort=True):
        monthly_rows.append({
            "forecast_month": month, "rhc_rows": len(part),
            "groundtruth_available": int(part.moh_groundtruth.notna().sum()),
            "moh_outbreaks": int(part.moh_groundtruth.sum()) if part.moh_groundtruth.notna().any() else pd.NA,
            "proposed_alerts": int(part.proposed_rhc_alert.sum()),
            "distribution_alerts": int(part.distribution_moh_alert.sum()),
            "current_alerts": int(part.current_rhc_alert.sum()),
        })
    pd.DataFrame(monthly_rows).to_csv(args.output.parent / "monthly_summary.csv", index=False)

    observed = available.observed_cases.to_numpy(float)
    predicted = available.rhc_prediction.to_numpy(float)
    count_metrics = {
        "n": len(available), "mae_cases": float(np.mean(np.abs(predicted - observed))),
        "rmse_cases": float(np.sqrt(np.mean(np.square(predicted - observed)))),
        "bias_cases": float(np.mean(predicted - observed)),
    }
    manifest = {
        "rows": len(output), "rhc_units": int(output.rhc_id.nunique()),
        "provinces": int(output.province.nunique()),
        "groundtruth_available_rows": len(available),
        "groundtruth_unavailable_rows": int(output.moh_groundtruth.isna().sum()),
        "groundtruth_rule": "observed RHC cases > RHC-specific prior-five-year same-month mean + 2 sample SD",
        "fair_comparison": "all alert methods scored against the same RHC-level MOH ground truth",
        "count_metrics_april_august": count_metrics,
    }
    (args.output.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": manifest, "metrics": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
