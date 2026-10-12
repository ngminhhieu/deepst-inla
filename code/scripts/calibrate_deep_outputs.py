#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from dengue_h3.evaluation import score_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--outbreak-weight", type=float, default=0.0)
    parser.add_argument(
        "--calibration-split",
        choices=("validation", "fit"),
        default="validation",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.features, dtype={"fcode": str})
    if not {"z_01", "z_02"}.issubset(frame.columns):
        raise ValueError("Expected location and scale columns z_01 and z_02")
    calibration = frame[
        frame["split"].eq("validation")
        if args.calibration_split == "validation"
        else frame["split"].isin(("train", "validation"))
    ]
    test = frame[frame["split"].eq("test")].copy()
    horizons = sorted(test["horizon"].unique())
    rng = np.random.default_rng(args.seed)
    calibrated = {}
    sample_parts = []

    for horizon in horizons:
        fit = calibration[calibration["horizon"].eq(horizon)]
        observed = np.log1p(fit["target_cases"].to_numpy(float))
        location = fit["z_01"].to_numpy(float)
        scale = np.maximum(fit["z_02"].to_numpy(float), 1e-4)
        outbreak = (
            fit["target_cases"].to_numpy(float)
            > fit["outbreak_threshold"].to_numpy(float)
        )
        weights = 1.0 + args.outbreak_weight * outbreak.astype(float)

        def objective(parameters: np.ndarray) -> float:
            intercept, slope, log_multiplier = parameters
            calibrated_scale = scale * np.exp(log_multiplier)
            residual = (observed - intercept - slope * location) / calibrated_scale
            nll = np.log(calibrated_scale) + 0.5 * np.square(residual)
            return float(np.average(nll, weights=weights))

        result = minimize(
            objective,
            x0=np.array([0.0, 1.0, 0.0]),
            method="L-BFGS-B",
            bounds=((-5.0, 5.0), (0.05, 3.0), (-2.0, 2.0)),
        )
        if not result.success:
            raise RuntimeError(f"Calibration failed at H{horizon}: {result.message}")
        intercept, slope, log_multiplier = result.x
        calibrated[str(int(horizon))] = {
            "intercept": float(intercept),
            "slope": float(slope),
            "scale_multiplier": float(np.exp(log_multiplier)),
            "validation_objective": float(result.fun),
        }

        selected = test["horizon"].eq(horizon)
        test_location = intercept + slope * test.loc[selected, "z_01"].to_numpy(float)
        test_scale = (
            test.loc[selected, "z_02"].to_numpy(float) * np.exp(log_multiplier)
        )
        latent_draws = rng.normal(
            test_location[:, None], test_scale[:, None],
            size=(selected.sum(), args.draws),
        )
        sample_parts.append(np.maximum(np.expm1(latent_draws), 0).astype(np.float32))

    origins = test["issue_date"].nunique()
    units = test["fcode"].nunique()
    samples = np.empty((origins, units, len(horizons), args.draws), dtype=np.float32)
    for horizon_index, (horizon, horizon_samples) in enumerate(zip(horizons, sample_parts)):
        samples[:, :, horizon_index, :] = horizon_samples.reshape(
            origins, units, args.draws
        )

    prediction_rows = []
    for horizon_index, horizon in enumerate(horizons):
        part = test[test["horizon"].eq(horizon)].copy().reset_index(drop=True)
        values = samples[:, :, horizon_index, :].reshape(-1, args.draws)
        part["model"] = args.model_label
        part["predicted_mean"] = values.mean(axis=1)
        part["predicted_median"] = np.median(values, axis=1)
        for level, lower_probability, upper_probability in (
            (50, 0.25, 0.75), (80, 0.10, 0.90), (95, 0.025, 0.975)
        ):
            part[f"lower_{level}"] = np.quantile(values, lower_probability, axis=1)
            part[f"upper_{level}"] = np.quantile(values, upper_probability, axis=1)
        part["outbreak_probability"] = (
            values > part["outbreak_threshold"].to_numpy(float)[:, None]
        ).mean(axis=1)
        prediction_rows.append(part)

    predictions = pd.concat(prediction_rows, ignore_index=True)
    prediction_columns = [
        "model", "fcode", "issue_date", "target_date", "horizon",
        "target_cases", "population", "predicted_mean", "predicted_median",
        "lower_50", "upper_50", "lower_80", "upper_80", "lower_95",
        "upper_95", "outbreak_threshold", "outbreak_probability",
    ]
    predictions = predictions[prediction_columns]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    score_predictions(predictions, samples).to_csv(
        args.output_dir / "metrics.csv", index=False
    )
    (args.output_dir / "calibration.json").write_text(
        json.dumps(
            {
                "model": args.model_label,
                "selection_split": args.calibration_split,
                "outbreak_weight": args.outbreak_weight,
                "parameters_by_horizon": calibrated,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
