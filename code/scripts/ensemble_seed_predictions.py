#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_h3.evaluation import score_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("result_dirs", type=Path, nargs="+")
    parser.add_argument("--model-label", default="GNN-SEED-ENSEMBLE")
    args = parser.parse_args()

    prediction_frames = [
        pd.read_csv(path / "predictions.csv", dtype={"fcode": str})
        for path in args.result_dirs
    ]
    identity_columns = [
        "fcode", "issue_date", "target_date", "horizon", "target_cases",
        "population", "outbreak_threshold",
    ]
    if any(
        not prediction_frames[0][identity_columns].equals(frame[identity_columns])
        for frame in prediction_frames[1:]
    ):
        raise ValueError("Seed prediction rows are not aligned")

    sample_arrays = [
        np.load(path / "predictive_samples.npz")["samples"]
        for path in args.result_dirs
    ]
    total_draws = sample_arrays[0].shape[-1]
    base_count, remainder = divmod(total_draws, len(sample_arrays))
    counts = [base_count + (index < remainder) for index in range(len(sample_arrays))]
    samples = np.concatenate(
        [array[..., :count] for array, count in zip(sample_arrays, counts)], axis=-1
    )

    base = prediction_frames[0].copy()
    base["model"] = args.model_label
    for horizon_index, horizon in enumerate(sorted(base["horizon"].unique())):
        selected = base["horizon"].eq(horizon)
        values = samples[:, :, horizon_index, :].reshape(-1, total_draws)
        base.loc[selected, "predicted_mean"] = values.mean(axis=1)
        base.loc[selected, "predicted_median"] = np.median(values, axis=1)
        for level, lower_probability, upper_probability in (
            (50, 0.25, 0.75), (80, 0.10, 0.90), (95, 0.025, 0.975)
        ):
            base.loc[selected, f"lower_{level}"] = np.quantile(
                values, lower_probability, axis=1
            )
            base.loc[selected, f"upper_{level}"] = np.quantile(
                values, upper_probability, axis=1
            )
        base.loc[selected, "outbreak_probability"] = (
            values
            > base.loc[selected, "outbreak_threshold"].to_numpy(float)[:, None]
        ).mean(axis=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base.to_csv(args.output_dir / "predictions.csv", index=False)
    np.savez_compressed(args.output_dir / "predictive_samples.npz", samples=samples)
    score_predictions(base, samples).to_csv(
        args.output_dir / "metrics.csv", index=False
    )


if __name__ == "__main__":
    main()
