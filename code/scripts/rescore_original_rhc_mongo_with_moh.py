#!/usr/bin/env python3
"""Re-score MongoDB's original DeepAR-NB RHC forecasts with MOH thresholds."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson


def exceedance_probability(mean: float, sd: float, threshold: float) -> float:
    """P(X > threshold) for the stored DeepAR negative-binomial forecast."""
    if not all(math.isfinite(value) for value in (mean, sd, threshold)):
        return float("nan")
    mean, variance = max(mean, 0.0), max(sd, 0.0) ** 2
    cutoff = math.floor(threshold)
    if mean <= 1e-12:
        return float(cutoff < 0)
    if variance > mean + 1e-8:
        size = mean * mean / (variance - mean)
        probability = size / (size + mean)
        return float(nbinom.sf(cutoff, size, probability))
    return float(poisson.sf(cutoff, mean))


def binary_metrics(labels, predictions):
    labels, predictions = np.asarray(labels, bool), np.asarray(predictions, bool)
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
    parser.add_argument("--mongo-export", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--province-lookup", type=Path, required=True)
    parser.add_argument("--proposed-predictions", type=Path, required=True)
    parser.add_argument("--proposed-samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    documents = json.loads(args.mongo_export.read_text(encoding="utf-8"))
    rows = []
    for doc in documents:
        rhc_id = str(doc["fcode"])
        for month in range(4, 12):
            month_doc = (doc.get("months") or {}).get(str(month), {})
            horizon = (month_doc.get("horizons") or {}).get("3", {})
            if not horizon:
                continue
            mean = float(horizon.get("pred_cases_mean", np.nan))
            sd = float(horizon.get("pred_cases_std", np.nan))
            outbreak_threshold = float(month_doc.get("epi_threshold_moh", np.nan))
            warning_threshold = float(month_doc.get("alert_threshold_moh", np.nan))
            rows.append({
                "forecast_month": f"2026-{month:02d}", "rhc_id": rhc_id,
                "issue_month": str((pd.Timestamp(2026, month, 1) - pd.DateOffset(months=3)).to_period("M")),
                "original_prediction": mean, "original_prediction_sd": sd,
                "original_75ci_lower": (horizon.get("pred_cases_75ci") or [np.nan, np.nan])[0],
                "original_75ci_upper": (horizon.get("pred_cases_75ci") or [np.nan, np.nan])[1],
                "original_95ci_lower": (horizon.get("pred_cases_95ci") or [np.nan, np.nan])[0],
                "original_95ci_upper": (horizon.get("pred_cases_95ci") or [np.nan, np.nan])[1],
                "moh_warning_threshold_rhc": warning_threshold,
                "moh_outbreak_threshold_rhc": outbreak_threshold,
                "original_moh_outbreak_probability": exceedance_probability(mean, sd, outbreak_threshold),
                "original_stored_outbreak_probability": horizon.get("prob_outbreak"),
                "original_stored_epi_threshold": horizon.get("epi_threshold"),
                "current_intervention_flag": horizon.get("intervention_flag"),
            })
    output = pd.DataFrame(rows)
    if len(output) != 128 * 8:
        raise ValueError(f"Expected 1024 RHC-month forecasts, found {len(output)}")
    output["original_moh_outbreak_alert"] = pd.array(
        output.original_moh_outbreak_probability >= 0.50, dtype="Int64"
    )
    output["original_stored_outbreak_alert"] = pd.array(
        pd.to_numeric(output.original_stored_outbreak_probability, errors="coerce") >= 0.50,
        dtype="Int64",
    )

    observations = pd.read_csv(
        args.observations, usecols=["fcode", "year", "month", "obs_dengue_cases"],
        dtype={"fcode": str},
    )
    observations["forecast_month"] = (
        observations.year.astype(str) + "-" + observations.month.astype(str).str.zfill(2)
    )
    observations = observations.rename(columns={"fcode": "rhc_id", "obs_dengue_cases": "observed_cases"})
    output = output.merge(
        observations[["rhc_id", "forecast_month", "observed_cases"]],
        on=["rhc_id", "forecast_month"], how="left", validate="one_to_one",
    )
    output["moh_groundtruth"] = pd.array([
        pd.NA if pd.isna(observed) else int(float(observed) > float(threshold))
        for observed, threshold in zip(output.observed_cases, output.moh_outbreak_threshold_rhc)
    ], dtype="Int64")

    proposed = pd.read_csv(args.proposed_predictions, dtype={"fcode": str})
    proposed["forecast_month"] = pd.to_datetime(proposed.target_date).dt.strftime("%Y-%m")
    proposed = proposed.rename(columns={"fcode": "rhc_id", "predicted_mean": "proposed_prediction"})
    samples = np.load(args.proposed_samples)["samples"]
    if len(proposed) != len(samples):
        raise ValueError("Proposed prediction/sample mismatch")
    proposed["_sample_row"] = np.arange(len(proposed))
    output = output.merge(
        proposed[["rhc_id", "forecast_month", "proposed_prediction", "_sample_row"]],
        on=["rhc_id", "forecast_month"], how="left", validate="one_to_one",
    )
    positions = output._sample_row.astype(int).to_numpy()
    proposed_values = samples[positions]
    output["proposed_moh_outbreak_probability"] = np.mean(
        proposed_values > output.moh_outbreak_threshold_rhc.to_numpy(float)[:, None], axis=1
    )
    output["proposed_moh_outbreak_alert"] = pd.array(
        output.proposed_moh_outbreak_probability >= 0.50, dtype="Int64"
    )
    province = pd.read_csv(args.province_lookup, dtype={"fcode": str, "province_code": str})
    province = province.rename(columns={"fcode": "rhc_id"})
    output = output.merge(province, on="rhc_id", how="left", validate="many_to_one")
    output = output.drop(columns="_sample_row").sort_values(["forecast_month", "rhc_id"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "original_rhc_predictions_rescored_moh.csv", index=False, na_rep="NA")
    available = output.loc[output.moh_groundtruth.notna()].copy()
    metric_rows = []
    for model, column in [
        ("Original DeepAR-NB re-scored with MOH", "original_moh_outbreak_alert"),
        ("Proposed hybrid distribution re-scored with MOH", "proposed_moh_outbreak_alert"),
        ("Original stored outbreak probability", "original_stored_outbreak_alert"),
        ("Current intervention flag", "current_intervention_flag"),
    ]:
        prediction = pd.to_numeric(available[column], errors="coerce")
        valid = prediction.notna()
        metric_rows.append({
            "model": model,
            **binary_metrics(
                available.loc[valid, "moh_groundtruth"].to_numpy(int),
                prediction.loc[valid].to_numpy(int),
            ),
        })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "moh_rescored_metrics.csv", index=False)

    count_rows = []
    observed = available.observed_cases.to_numpy(float)
    for model, column in [
        ("Original DeepAR-NB", "original_prediction"),
        ("Proposed hybrid", "proposed_prediction"),
    ]:
        predicted = available[column].to_numpy(float)
        count_rows.append({
            "model": model, "n": len(available),
            "mae_cases": float(np.mean(np.abs(predicted - observed))),
            "rmse_cases": float(np.sqrt(np.mean(np.square(predicted - observed)))),
            "bias_cases": float(np.mean(predicted - observed)),
        })
    pd.DataFrame(count_rows).to_csv(args.output_dir / "count_metrics.csv", index=False)

    stored_probability = pd.to_numeric(output.original_stored_outbreak_probability, errors="coerce")
    stored_threshold = pd.to_numeric(output.original_stored_epi_threshold, errors="coerce")
    reconstructed_stored = np.asarray([
        exceedance_probability(mean, sd, threshold)
        for mean, sd, threshold in zip(output.original_prediction, output.original_prediction_sd, stored_threshold)
    ])
    validation = {
        "stored_probability_rows": int(stored_probability.notna().sum()),
        "mean_absolute_probability_reconstruction_error": float(np.nanmean(
            np.abs(reconstructed_stored - stored_probability.to_numpy(float))
        )),
        "distribution_assumption": "Negative binomial reconstructed from stored mean and standard deviation; Poisson fallback when variance <= mean",
        "outbreak_rule": "observed/predictive cases > epi_threshold_moh",
        "alert_rule": "MOH outbreak probability >= 0.50",
        "evaluation_months": ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08"],
        "forecast_only_months": ["2026-09", "2026-10", "2026-11"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
