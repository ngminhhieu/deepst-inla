#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_h3.evaluation import _crps, outbreak_classification_metrics, score_predictions


IDENTITY_COLUMNS = [
    "fcode", "issue_date", "target_date", "horizon", "target_cases",
    "population", "outbreak_threshold",
]


def load_split(result_dir: Path, split: str):
    prefix = "validation_" if split == "validation" else ""
    predictions = pd.read_csv(
        result_dir / f"{prefix}predictions.csv", dtype={"fcode": str}
    )
    samples = np.load(result_dir / f"{prefix}predictive_samples.npz")["samples"]
    return predictions, samples


def validation_grid(low_dir: Path, high_dir: Path):
    low_predictions, low_samples = load_split(low_dir, "validation")
    high_predictions, high_samples = load_split(high_dir, "validation")
    if not low_predictions[IDENTITY_COLUMNS].equals(
        high_predictions[IDENTITY_COLUMNS]
    ):
        raise ValueError("Validation expert rows are not aligned")
    candidates = np.concatenate((np.arange(0.05, 0.55, 0.05), [1.01]))
    grids = {}
    selected_crps, selected_f1 = {}, {}
    for horizon_index, horizon in enumerate(sorted(low_predictions.horizon.unique())):
        selected = low_predictions.horizon.eq(horizon)
        part = low_predictions[selected].reset_index(drop=True)
        low_values = low_samples[:, :, horizon_index, :].reshape(-1, low_samples.shape[-1])
        high_values = high_samples[:, :, horizon_index, :].reshape(-1, high_samples.shape[-1])
        observed = part.target_cases.to_numpy(float)
        population = part.population.to_numpy(float)
        observed_incidence = observed / population * 100_000
        scores = []
        for threshold in candidates:
            gate = part.outbreak_probability.to_numpy(float) >= threshold
            values = np.where(gate[:, None], high_values, low_values)
            incidence = values / population[:, None] * 100_000
            probability = (
                values > part.outbreak_threshold.to_numpy(float)[:, None]
            ).mean(axis=1)
            classification = outbreak_classification_metrics(
                observed, part.outbreak_threshold.to_numpy(float), probability
            )
            scores.append(
                {
                    "threshold": float(threshold),
                    "gate_fraction": float(gate.mean()),
                    "crps": float(_crps(observed_incidence, incidence).mean()),
                    "f1": float(classification["outbreak_f1"]),
                    "precision": float(classification["outbreak_precision"]),
                    "specificity": float(classification["outbreak_specificity"]),
                    "recall": float(classification["outbreak_recall"]),
                    "sensitivity": float(classification["outbreak_sensitivity"]),
                }
            )
        best_crps = min(scores, key=lambda row: row["crps"])
        best_f1 = max(scores, key=lambda row: (row["f1"], -row["crps"]))
        grids[str(int(horizon))] = scores
        selected_crps[str(int(horizon))] = best_crps["threshold"]
        selected_f1[str(int(horizon))] = best_f1["threshold"]
    return selected_crps, selected_f1, grids


def apply_policy(
    low_dir: Path,
    high_dir: Path,
    output_dir: Path,
    label: str,
    thresholds: dict[str, float],
    split: str = "test",
):
    low_predictions, low_samples = load_split(low_dir, split)
    high_predictions, high_samples = load_split(high_dir, split)
    if not low_predictions[IDENTITY_COLUMNS].equals(high_predictions[IDENTITY_COLUMNS]):
        raise ValueError("Test expert rows are not aligned")
    samples = np.empty_like(low_samples)
    predictions = low_predictions.copy()
    predictions["model"] = label
    gate_fractions = {}
    for horizon_index, horizon in enumerate(sorted(predictions.horizon.unique())):
        selected = predictions.horizon.eq(horizon)
        part = predictions[selected].reset_index(drop=True)
        gate = (
            part.outbreak_probability.to_numpy(float)
            >= thresholds[str(int(horizon))]
        )
        low_values = low_samples[:, :, horizon_index, :].reshape(-1, low_samples.shape[-1])
        high_values = high_samples[:, :, horizon_index, :].reshape(-1, high_samples.shape[-1])
        values = np.where(gate[:, None], high_values, low_values)
        samples[:, :, horizon_index, :] = values.reshape(
            low_samples.shape[0], low_samples.shape[1], low_samples.shape[-1]
        )
        predictions.loc[selected, "predicted_mean"] = values.mean(axis=1)
        predictions.loc[selected, "predicted_median"] = np.median(values, axis=1)
        for level, lower_probability, upper_probability in (
            (50, 0.25, 0.75), (80, 0.10, 0.90), (95, 0.025, 0.975)
        ):
            predictions.loc[selected, f"lower_{level}"] = np.quantile(
                values, lower_probability, axis=1
            )
            predictions.loc[selected, f"upper_{level}"] = np.quantile(
                values, upper_probability, axis=1
            )
        predictions.loc[selected, "outbreak_probability"] = (
            values
            > part.outbreak_threshold.to_numpy(float)[:, None]
        ).mean(axis=1)
        gate_fractions[str(int(horizon))] = float(gate.mean())
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "validation_" if split == "validation" else ""
    predictions.to_csv(output_dir / f"{prefix}predictions.csv", index=False)
    np.savez_compressed(
        output_dir / f"{prefix}predictive_samples.npz", samples=samples
    )
    score_predictions(predictions, samples).to_csv(
        output_dir / f"{prefix}metrics.csv", index=False
    )
    return gate_fractions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("low_result_dir", type=Path)
    parser.add_argument("high_result_dir", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    crps_thresholds, f1_thresholds, grid = validation_grid(
        args.low_result_dir, args.high_result_dir
    )
    policies = {
        "MOE-GNN-CRPS": crps_thresholds,
        "MOE-GNN-F1": f1_thresholds,
    }
    test_gate_fractions = {}
    for label, thresholds in policies.items():
        apply_policy(
            args.low_result_dir,
            args.high_result_dir,
            args.output_root / label,
            label,
            thresholds,
            split="validation",
        )
        test_gate_fractions[label] = apply_policy(
            args.low_result_dir,
            args.high_result_dir,
            args.output_root / label,
            label,
            thresholds,
        )
    (args.output_root / "selection.json").write_text(
        json.dumps(
            {
                "selection_split": "validation",
                "gate_signal": "low_expert_outbreak_probability",
                "high_expert": args.high_result_dir.name,
                "low_expert": args.low_result_dir.name,
                "thresholds": policies,
                "test_gate_fractions": test_gate_fractions,
                "validation_grid": grid,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
