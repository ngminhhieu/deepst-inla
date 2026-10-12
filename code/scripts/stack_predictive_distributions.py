#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_h3.evaluation import (
    _crps,
    outbreak_classification_metrics,
    score_predictions,
)
from ensemble_model_predictions import aligned_samples


IDENTITY = [
    "fcode",
    "issue_date",
    "target_date",
    "horizon",
    "target_cases",
    "population",
    "outbreak_threshold",
]


def predictions_from_horizon_samples(
    template: pd.DataFrame,
    samples: np.ndarray,
    model_label: str,
) -> pd.DataFrame:
    """Build predictions when exact target windows shift issue dates by horizon."""
    predictions = template.copy()
    predictions["model"] = model_label
    horizons = list(dict.fromkeys(template["horizon"].astype(int)))
    for horizon_index, horizon in enumerate(horizons):
        selected = template["horizon"].eq(horizon)
        part = template.loc[selected].reset_index(drop=True)
        values = samples[:, :, horizon_index, :].reshape(-1, samples.shape[-1])
        if len(values) != len(part):
            raise ValueError(f"Sample rows do not align for H{horizon}")
        predictions.loc[selected, "predicted_mean"] = values.mean(axis=1)
        predictions.loc[selected, "predicted_median"] = np.median(values, axis=1)
        for level, lower, upper in (
            (50, 0.25, 0.75),
            (80, 0.10, 0.90),
            (95, 0.025, 0.975),
        ):
            predictions.loc[selected, f"lower_{level}"] = np.quantile(
                values, lower, axis=1
            )
            predictions.loc[selected, f"upper_{level}"] = np.quantile(
                values, upper, axis=1
            )
        predictions.loc[selected, "outbreak_probability"] = (
            values > part["outbreak_threshold"].to_numpy(float)[:, None]
        ).mean(axis=1)
    return predictions


def load_split(directory: Path, validation: bool) -> tuple[pd.DataFrame, np.ndarray]:
    prefix = "validation_" if validation else ""
    frame = pd.read_csv(
        directory / f"{prefix}predictions.csv",
        dtype={"fcode": str},
    )
    samples = np.load(directory / f"{prefix}predictive_samples.npz")["samples"]
    return frame, samples


def integer_compositions(total: int, members: int):
    if members == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for remainder in integer_compositions(total - first, members - 1):
            yield (first, *remainder)


def mix_draws(arrays: list[np.ndarray], counts: tuple[int, ...]) -> np.ndarray:
    pieces = [
        array[..., :count]
        for array, count in zip(arrays, counts)
        if count > 0
    ]
    return np.concatenate(pieces, axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--members", type=Path, nargs="+", required=True)
    parser.add_argument("--selection-draws", type=int, default=100)
    parser.add_argument("--crps-tolerance", type=float, default=0.03)
    parser.add_argument("--minimum-precision", type=float, default=0.50)
    args = parser.parse_args()

    validation_runs = [load_split(path, True) for path in args.members]
    test_runs = [load_split(path, False) for path in args.members]
    validation_template = validation_runs[0][0]
    test_template = test_runs[0][0]
    for predictions, _ in validation_runs[1:]:
        if not validation_template[IDENTITY].equals(predictions[IDENTITY]):
            raise ValueError("Validation stacking member rows are not aligned")
    for predictions, _ in test_runs[1:]:
        if not test_template[IDENTITY].equals(predictions[IDENTITY]):
            raise ValueError("Stacking member rows are not aligned")

    validation_arrays = [
        aligned_samples(validation_template, predictions, samples)
        for predictions, samples in validation_runs
    ]
    test_arrays = [
        aligned_samples(test_template, predictions, samples)
        for predictions, samples in test_runs
    ]
    horizons = list(dict.fromkeys(validation_template["horizon"].astype(int)))
    composition_units = 10
    compositions = list(integer_compositions(composition_units, len(args.members)))
    threshold_candidates = np.round(np.arange(0.05, 0.81, 0.01), 2)
    selected: dict[int, dict[str, object]] = {}
    validation_grid: dict[str, list[dict[str, object]]] = {}

    for horizon_index, horizon in enumerate(horizons):
        part = validation_template[
            validation_template["horizon"].eq(horizon)
        ].reset_index(drop=True)
        observed = part["target_cases"].to_numpy(float)
        population = part["population"].to_numpy(float)
        observed_incidence = observed / population * 100_000.0
        threshold_counts = part["outbreak_threshold"].to_numpy(float)
        rows = []
        for composition in compositions:
            draw_counts = tuple(
                int(round(args.selection_draws * value / composition_units))
                for value in composition
            )
            difference = args.selection_draws - sum(draw_counts)
            draw_counts = (
                draw_counts[0] + difference,
                *draw_counts[1:],
            )
            values = mix_draws(
                [
                    array[:, :, horizon_index, : args.selection_draws]
                    for array in validation_arrays
                ],
                draw_counts,
            ).reshape(len(part), args.selection_draws)
            incidence = values / population[:, None] * 100_000.0
            crps = float(_crps(observed_incidence, incidence).mean())
            probability = (values > threshold_counts[:, None]).mean(axis=1)
            best_alert = None
            for threshold in threshold_candidates:
                metrics = outbreak_classification_metrics(
                    observed,
                    threshold_counts,
                    probability,
                    decision_threshold=float(threshold),
                )
                candidate = {
                    "threshold": float(threshold),
                    "f1": float(metrics["outbreak_f1"]),
                    "precision": float(metrics["outbreak_precision"]),
                    "recall": float(metrics["outbreak_recall"]),
                }
                if candidate["precision"] < args.minimum_precision:
                    continue
                if best_alert is None or (
                    candidate["f1"],
                    candidate["recall"],
                ) > (
                    best_alert["f1"],
                    best_alert["recall"],
                ):
                    best_alert = candidate
            if best_alert is None:
                best_alert = {
                    "threshold": 0.5,
                    "f1": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                }
            rows.append(
                {
                    "weights": [
                        value / composition_units for value in composition
                    ],
                    "crps": crps,
                    **best_alert,
                }
            )
        best_crps = min(row["crps"] for row in rows)
        eligible = [
            row
            for row in rows
            if row["crps"] <= best_crps * (1.0 + args.crps_tolerance)
        ]
        winner = max(
            eligible,
            key=lambda row: (
                row["f1"],
                row["recall"],
                -row["crps"],
            ),
        )
        selected[int(horizon)] = winner
        validation_grid[str(int(horizon))] = rows

    total_draws = test_arrays[0].shape[-1]
    mixed = np.empty_like(test_arrays[0], dtype=np.float32)
    thresholds = {}
    for horizon_index, horizon in enumerate(horizons):
        weights = selected[int(horizon)]["weights"]
        counts = tuple(
            int(round(total_draws * weight))
            for weight in weights
        )
        difference = total_draws - sum(counts)
        counts = (counts[0] + difference, *counts[1:])
        mixed[:, :, horizon_index] = mix_draws(
            [array[:, :, horizon_index] for array in test_arrays],
            counts,
        )
        thresholds[int(horizon)] = float(selected[int(horizon)]["threshold"])

    predictions = predictions_from_horizon_samples(
        test_template,
        mixed,
        args.model_label,
    )
    predictions["outbreak_decision_threshold"] = predictions["horizon"].map(
        thresholds
    )
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    np.savez_compressed(args.output / "predictive_samples.npz", samples=mixed)
    score_predictions(
        predictions,
        mixed,
        decision_threshold=thresholds,
    ).to_csv(args.output / "metrics.csv", index=False)
    (args.output / "stacking_selection.json").write_text(
        json.dumps(
            {
                "model": args.model_label,
                "members": [str(path) for path in args.members],
                "selection_split": "validation",
                "selection_draws": args.selection_draws,
                "crps_tolerance": args.crps_tolerance,
                "minimum_precision": args.minimum_precision,
                "selected_by_horizon": selected,
                "validation_grid": validation_grid,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
