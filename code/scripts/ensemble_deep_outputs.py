#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_h3.evaluation import _crps, score_predictions


KEYS = ["split", "fcode", "issue_date", "target_date", "horizon"]


def draw_lognormal(frame: pd.DataFrame, draws: int, rng: np.random.Generator) -> np.ndarray:
    latent = rng.normal(
        frame["z_01"].to_numpy(float)[:, None],
        frame["z_02"].to_numpy(float)[:, None],
        size=(len(frame), draws),
    )
    return np.maximum(np.expm1(latent), 0).astype(np.float32)


def mixture(left: np.ndarray, right: np.ndarray, weight: float) -> np.ndarray:
    left_count = int(round(left.shape[1] * weight))
    return np.concatenate(
        (left[:, :left_count], right[:, : left.shape[1] - left_count]), axis=1
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left_features", type=Path)
    parser.add_argument("right_features", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model-label", default="GNN-TCN-GNN-ENSEMBLE")
    parser.add_argument("--selection-draws", type=int, default=200)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    left = pd.read_csv(args.left_features, dtype={"fcode": str})
    right = pd.read_csv(args.right_features, dtype={"fcode": str})
    required = KEYS + ["target_cases", "population", "outbreak_threshold"]
    if not left[required].equals(right[required]):
        raise ValueError("Deep feature rows are not aligned")

    weights = np.linspace(0, 1, 11)
    selected_weights: dict[str, float] = {}
    validation_scores: dict[str, dict[str, float]] = {}
    for horizon in sorted(left["horizon"].unique()):
        selected = left["split"].eq("validation") & left["horizon"].eq(horizon)
        left_part, right_part = left[selected], right[selected]
        left_draws = draw_lognormal(
            left_part, args.selection_draws, np.random.default_rng(args.seed + int(horizon))
        )
        right_draws = draw_lognormal(
            right_part,
            args.selection_draws,
            np.random.default_rng(args.seed + 100 + int(horizon)),
        )
        observed = left_part["target_cases"].to_numpy(float)
        population = left_part["population"].to_numpy(float)
        observed_incidence = observed / population * 100_000
        scores = {}
        for weight in weights:
            combined = mixture(left_draws, right_draws, float(weight))
            incidence = combined / population[:, None] * 100_000
            scores[f"{weight:.1f}"] = float(_crps(observed_incidence, incidence).mean())
        best = min(scores, key=scores.get)
        selected_weights[str(int(horizon))] = float(best)
        validation_scores[str(int(horizon))] = scores

    test = left[left["split"].eq("test")].copy()
    horizons = sorted(test["horizon"].unique())
    origins, units = test["issue_date"].nunique(), test["fcode"].nunique()
    samples = np.empty((origins, units, len(horizons), args.draws), dtype=np.float32)
    prediction_rows = []
    for horizon_index, horizon in enumerate(horizons):
        selected = left["split"].eq("test") & left["horizon"].eq(horizon)
        left_part, right_part = left[selected], right[selected]
        left_draws = draw_lognormal(
            left_part, args.draws, np.random.default_rng(args.seed + int(horizon))
        )
        right_draws = draw_lognormal(
            right_part, args.draws, np.random.default_rng(args.seed + 100 + int(horizon))
        )
        values = mixture(left_draws, right_draws, selected_weights[str(int(horizon))])
        samples[:, :, horizon_index, :] = values.reshape(origins, units, args.draws)
        part = left_part.copy().reset_index(drop=True)
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
    columns = [
        "model", "fcode", "issue_date", "target_date", "horizon",
        "target_cases", "population", "predicted_mean", "predicted_median",
        "lower_50", "upper_50", "lower_80", "upper_80", "lower_95",
        "upper_95", "outbreak_threshold", "outbreak_probability",
    ]
    predictions = predictions[columns]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    score_predictions(predictions, samples).to_csv(
        args.output_dir / "metrics.csv", index=False
    )
    (args.output_dir / "selection.json").write_text(
        json.dumps(
            {
                "model": args.model_label,
                "selection_split": "validation",
                "left_model": left["model"].iloc[0] if "model" in left else "GNN",
                "right_model": right["model"].iloc[0] if "model" in right else "TCN-GNN",
                "left_weights_by_horizon": selected_weights,
                "validation_crps_grid": validation_scores,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
