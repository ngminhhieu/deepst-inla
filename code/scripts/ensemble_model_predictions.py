#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_h3.evaluation import score_predictions


IDENTITY = [
    "fcode",
    "issue_date",
    "target_date",
    "horizon",
    "target_cases",
    "population",
    "outbreak_threshold",
]


def load_run(directory: Path) -> tuple[pd.DataFrame, np.ndarray]:
    predictions = pd.read_csv(directory / "predictions.csv", dtype={"fcode": str})
    samples = np.load(directory / "predictive_samples.npz")["samples"]
    origins = predictions["issue_date"].nunique()
    units = predictions["fcode"].nunique()
    horizons = predictions["horizon"].nunique()
    expected = (origins, units, horizons, samples.shape[-1])
    if samples.shape[:3] != expected[:3]:
        raise ValueError(f"{directory} sample shape {samples.shape} != {expected}")
    return predictions, samples


def aligned_samples(template: pd.DataFrame, predictions: pd.DataFrame, samples: np.ndarray) -> np.ndarray:
    origin_values = list(dict.fromkeys(template["issue_date"].tolist()))
    unit_values = list(dict.fromkeys(template["fcode"].tolist()))
    horizon_values = list(dict.fromkeys(template["horizon"].tolist()))
    source_origin = list(dict.fromkeys(predictions["issue_date"].tolist()))
    source_unit = list(dict.fromkeys(predictions["fcode"].tolist()))
    source_horizon = list(dict.fromkeys(predictions["horizon"].tolist()))
    if (
        origin_values == source_origin
        and unit_values == source_unit
        and horizon_values == source_horizon
    ):
        return samples
    origin_index = {value: index for index, value in enumerate(source_origin)}
    unit_index = {value: index for index, value in enumerate(source_unit)}
    horizon_index = {value: index for index, value in enumerate(source_horizon)}
    output = np.empty_like(samples)
    for new_origin, origin in enumerate(origin_values):
        for new_unit, unit in enumerate(unit_values):
            for new_horizon, horizon in enumerate(horizon_values):
                output[new_origin, new_unit, new_horizon] = samples[
                    origin_index[origin], unit_index[unit], horizon_index[horizon]
                ]
    return output


def predictions_from_samples(
    template: pd.DataFrame,
    samples: np.ndarray,
    model_label: str,
) -> pd.DataFrame:
    rows = []
    origin_values = list(dict.fromkeys(template["issue_date"].tolist()))
    unit_values = list(dict.fromkeys(template["fcode"].tolist()))
    horizon_values = list(dict.fromkeys(template["horizon"].tolist()))
    lookup = template.set_index(["issue_date", "fcode", "horizon"])
    for origin_index, origin in enumerate(origin_values):
        for unit_index, unit in enumerate(unit_values):
            for horizon_index, horizon in enumerate(horizon_values):
                values = samples[origin_index, unit_index, horizon_index]
                row = lookup.loc[(origin, unit, horizon)]
                threshold = float(row["outbreak_threshold"])
                probability = float((values > threshold).mean())
                rows.append(
                    {
                        "model": model_label,
                        "fcode": unit,
                        "issue_date": origin,
                        "target_date": row["target_date"],
                        "horizon": int(horizon),
                        "target_cases": float(row["target_cases"]),
                        "population": float(row["population"]),
                        "predicted_mean": float(values.mean()),
                        "predicted_median": float(np.median(values)),
                        "lower_50": float(np.quantile(values, 0.25)),
                        "upper_50": float(np.quantile(values, 0.75)),
                        "lower_80": float(np.quantile(values, 0.10)),
                        "upper_80": float(np.quantile(values, 0.90)),
                        "lower_95": float(np.quantile(values, 0.025)),
                        "upper_95": float(np.quantile(values, 0.975)),
                        "outbreak_threshold": threshold,
                        "outbreak_probability": probability,
                        "outbreak_probability_samples": probability,
                        "alert_probability_source": "samples",
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--members", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    loaded = [load_run(path) for path in args.members]
    template = loaded[0][0]
    aligned = [
        aligned_samples(template, predictions, samples)
        for predictions, samples in loaded
    ]
    total_draws = aligned[0].shape[-1]
    base_count, remainder = divmod(total_draws, len(aligned))
    draw_counts = [
        base_count + (member_index < remainder)
        for member_index in range(len(aligned))
    ]
    mixed = np.concatenate(
        [
            samples[..., :draw_count]
            for samples, draw_count in zip(aligned, draw_counts)
        ],
        axis=-1,
    )
    predictions = predictions_from_samples(template, mixed, args.model_label)
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    np.savez_compressed(args.output / "predictive_samples.npz", samples=mixed)
    score_predictions(predictions, mixed).to_csv(args.output / "metrics.csv", index=False)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "model": args.model_label,
                "members": [str(path) for path in args.members],
                "combination": "equal-weight mixture distribution",
                "draw_counts": draw_counts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
