#!/usr/bin/env python3
"""Validation-locked outbreak-definition sensitivity analysis.

Alternative local thresholds are computed causally from the preceding five
observations for the same commune and calendar month. Probability calibration
and alert cutoffs are fitted on validation rows only, then frozen for test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["fcode", "target_date", "horizon"]
DEFINITIONS = ("mean_plus_1.5sd", "moh_audited", "mean_plus_2.5sd", "percentile_90")


def historical_thresholds(monthly_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(monthly_path, dtype={"fcode": str})
    frame["target_date"] = pd.to_datetime(
        {"year": frame["year"], "month": frame["month"], "day": 1}
    )
    frame = frame.sort_values(["fcode", "target_date"])
    rows = []
    for fcode, part in frame.groupby("fcode", sort=False):
        part = part.reset_index(drop=True)
        dates = pd.to_datetime(part["target_date"])
        cases = part["obs_dengue_cases"].to_numpy(float)
        for index, date in enumerate(dates):
            candidates = np.flatnonzero(
                (dates.dt.month.to_numpy() == date.month)
                & (dates.to_numpy() < np.datetime64(date))
                & (dates.dt.year.to_numpy() >= date.year - 5)
            )
            history = cases[candidates[-5:]]
            if len(history) < 2:
                continue
            mean = float(history.mean())
            sd = float(history.std(ddof=1))
            rows.append(
                {
                    "fcode": str(fcode),
                    "target_date": date,
                    "mean_plus_1.5sd": mean + 1.5 * sd,
                    "mean_plus_2sd": mean + 2.0 * sd,
                    "mean_plus_2.5sd": mean + 2.5 * sd,
                    "percentile_90": float(np.quantile(history, 0.90)),
                }
            )
    return pd.DataFrame(rows)


def load_cube(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        return archive["samples"]


def add_thresholds(
    frame: pd.DataFrame, thresholds: pd.DataFrame, moh: pd.DataFrame
) -> pd.DataFrame:
    output = frame.copy()
    output["fcode"] = output["fcode"].astype(str)
    output["target_date"] = pd.to_datetime(output["target_date"])
    output = output.merge(
        thresholds, on=["fcode", "target_date"], how="left", validate="many_to_one"
    )
    output = output.merge(
        moh, on=["fcode", "target_date"], how="left", validate="many_to_one"
    )
    if output[list(DEFINITIONS)].isna().any().any():
        raise ValueError("Threshold history does not cover every forecast row")
    return output


def fit_platt(probability: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-5, 1 - 1e-5)
    x = np.log(clipped / (1 - clipped))
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = np.array([0.0, 1.0])
    penalty = np.diag([0.0, 1e-3])
    for _ in range(50):
        linear = np.clip(design @ coefficients, -30, 30)
        fitted = 1 / (1 + np.exp(-linear))
        weight = np.clip(fitted * (1 - fitted), 1e-6, None)
        gradient = design.T @ (fitted - labels) + penalty @ coefficients
        hessian = design.T @ (weight[:, None] * design) + penalty
        update = np.linalg.solve(hessian, gradient)
        coefficients -= update
        if np.max(np.abs(update)) < 1e-8:
            break
    return float(coefficients[0]), float(coefficients[1])


def apply_platt(probability: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    clipped = np.clip(probability, 1e-5, 1 - 1e-5)
    logit = np.log(clipped / (1 - clipped))
    return 1 / (1 + np.exp(-np.clip(intercept + slope * logit, -30, 30)))


def classification(labels: np.ndarray, probability: np.ndarray, cutoff: float) -> dict[str, float | int]:
    alert = probability >= cutoff
    tp = int(np.sum(labels & alert))
    fp = int(np.sum(~labels & alert))
    fn = int(np.sum(labels & ~alert))
    tn = int(np.sum(~labels & ~alert))
    precision = tp / (tp + fp) if tp + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "observed_positive": int(labels.sum()),
        "predicted_positive": int(alert.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "f1": f1,
    }


def select_cutoff(labels: np.ndarray, probability: np.ndarray) -> tuple[float, dict[str, float | int]]:
    candidates = []
    for cutoff in np.arange(0.01, 1.0, 0.01):
        scores = classification(labels, probability, float(cutoff))
        candidates.append((scores["f1"], scores["specificity"], float(cutoff), scores))
    _, _, cutoff, scores = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    return cutoff, scores


def horizon_values(frame: pd.DataFrame, cube: np.ndarray, horizon: int) -> tuple[pd.DataFrame, np.ndarray]:
    part = frame[frame["horizon"].eq(horizon)].reset_index(drop=True)
    values = cube[:, :, horizon - 1, :].reshape(-1, cube.shape[-1])
    if len(part) != len(values):
        raise ValueError(f"H{horizon}: prediction/sample row mismatch")
    return part, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monthly", type=Path, required=True)
    parser.add_argument("--moh-thresholds", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--validation-samples", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--test-samples", type=Path, required=True)
    parser.add_argument("--model", default="TCN-GNN")
    parser.add_argument("--out", type=Path, default=Path("results/pending/threshold_sensitivity.csv"))
    args = parser.parse_args()

    thresholds = historical_thresholds(args.monthly)
    moh = pd.read_csv(args.moh_thresholds, dtype={"fcode": str})
    moh["target_date"] = pd.to_datetime(moh["target_date"])
    moh = moh[["fcode", "target_date", "epi_threshold"]].rename(
        columns={"epi_threshold": "moh_audited"}
    )
    validation = add_thresholds(
        pd.read_csv(args.validation_predictions, dtype={"fcode": str}), thresholds, moh
    )
    test = add_thresholds(
        pd.read_csv(args.test_predictions, dtype={"fcode": str}), thresholds, moh
    )
    validation_cube = load_cube(args.validation_samples)
    test_cube = load_cube(args.test_samples)
    rows = []
    selections: dict[str, object] = {}
    for definition in DEFINITIONS:
        selections[definition] = {}
        for horizon in (1, 2, 3):
            valid_part, valid_values = horizon_values(validation, validation_cube, horizon)
            test_part, test_values = horizon_values(test, test_cube, horizon)
            valid_threshold = valid_part[definition].to_numpy(float)
            test_threshold = test_part[definition].to_numpy(float)
            valid_labels = valid_part["target_cases"].to_numpy(float) > valid_threshold
            test_labels = test_part["target_cases"].to_numpy(float) > test_threshold
            valid_raw = (valid_values > valid_threshold[:, None]).mean(axis=1)
            test_raw = (test_values > test_threshold[:, None]).mean(axis=1)
            intercept, slope = fit_platt(valid_raw, valid_labels.astype(float))
            valid_probability = apply_platt(valid_raw, intercept, slope)
            test_probability = apply_platt(test_raw, intercept, slope)
            cutoff, validation_scores = select_cutoff(valid_labels, valid_probability)
            test_scores = classification(test_labels, test_probability, cutoff)
            rows.append(
                {
                    "model": args.model,
                    "definition": definition,
                    "horizon": horizon,
                    "validation_cutoff": cutoff,
                    "platt_intercept": intercept,
                    "platt_slope": slope,
                    "validation_f1": validation_scores["f1"],
                    "test_brier": float(np.square(test_probability - test_labels).mean()),
                    **{f"test_{key}": value for key, value in test_scores.items()},
                }
            )
            selections[definition][str(horizon)] = {
                "platt_intercept": intercept,
                "platt_slope": slope,
                "validation_cutoff": cutoff,
                "validation_f1": validation_scores["f1"],
            }
    output = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    args.out.with_suffix(".selection.json").write_text(
        json.dumps(
            {
                "selection_split": "validation predictions supplied by caller",
                "history": "alternative definitions use the preceding five same-calendar-month observations; moh_audited uses the official outbreak-year exclusion/replacement rule",
                "percentile_method": "numpy linear 90th percentile",
                "cutoff_rule": "maximum validation F1; ties prefer specificity then higher cutoff",
                "selections": selections,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
