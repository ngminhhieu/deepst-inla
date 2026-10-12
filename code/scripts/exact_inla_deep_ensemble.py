#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_h3.evaluation import _crps, outbreak_classification_metrics, score_predictions
from ensemble_inla_deep import load_deep


KEYS = ["fcode", "issue_date", "target_date", "horizon"]


def load_exact_horizon(
    inla_dir: Path, deep_part: pd.DataFrame, horizon: int, draws: int
) -> np.ndarray:
    predictions = pd.read_csv(inla_dir / "predictions.csv", dtype={"fcode": str})
    predictions = predictions[predictions.horizon.eq(horizon)].reset_index(drop=True)
    shape = pd.read_csv(inla_dir / f"samples_H{horizon}_shape.csv").iloc[0]
    if int(shape["rows"]) != len(predictions) or int(shape["draws"]) != draws:
        raise ValueError(f"Unexpected exact INLA sample shape at H{horizon}")
    samples = np.fromfile(inla_dir / f"samples_H{horizon}.bin", dtype=np.float64)
    samples = samples.reshape((len(predictions), draws), order="F").astype(np.float32)
    source_index = pd.MultiIndex.from_frame(predictions[KEYS])
    target_index = pd.MultiIndex.from_frame(deep_part[KEYS])
    positions = source_index.get_indexer(target_index)
    if np.any(positions < 0):
        raise ValueError(f"INLA and deep rows do not align at H{horizon}")
    return samples[positions]


def metrics_for_values(part: pd.DataFrame, values: np.ndarray) -> dict[str, float]:
    observed = part.target_cases.to_numpy(float)
    population = part.population.to_numpy(float)
    incidence = values / population[:, None] * 100_000
    observed_incidence = observed / population * 100_000
    probability = (
        values > part.outbreak_threshold.to_numpy(float)[:, None]
    ).mean(axis=1)
    classification = outbreak_classification_metrics(
        observed, part.outbreak_threshold.to_numpy(float), probability
    )
    return {
        "crps": float(_crps(observed_incidence, incidence).mean()),
        "f1": float(classification["outbreak_f1"]),
        "precision": float(classification["outbreak_precision"]),
        "specificity": float(classification["outbreak_specificity"]),
        "recall": float(classification["outbreak_recall"]),
        "sensitivity": float(classification["outbreak_sensitivity"]),
    }


def select_policies(inla_dir: Path, deep_dir: Path, draws: int):
    deep, deep_samples = load_deep(deep_dir, "validation")
    soft_grid, gate_grid = {}, {}
    selected = {name: {} for name in (
        "INLA-GNN-EXACT-SOFT-CRPS", "INLA-GNN-EXACT-SOFT-F1",
        "INLA-GNN-EXACT-GATE-CRPS", "INLA-GNN-EXACT-GATE-F1",
    )}
    for horizon_index, horizon in enumerate(sorted(deep.horizon.unique())):
        mask = deep.horizon.eq(horizon)
        part = deep[mask].reset_index(drop=True)
        deep_values = deep_samples[:, :, horizon_index, :].reshape(-1, draws)
        inla_values = load_exact_horizon(inla_dir, part, int(horizon), draws)
        soft_scores = []
        for weight in np.linspace(0, 1, 11):
            deep_count = int(round(draws * weight))
            values = np.concatenate(
                (deep_values[:, :deep_count], inla_values[:, : draws - deep_count]),
                axis=1,
            )
            soft_scores.append({"parameter": float(weight), **metrics_for_values(part, values)})
        gate_scores = []
        for threshold in np.concatenate(([-0.01], np.arange(0.05, 0.55, 0.05), [1.01])):
            gate = part.outbreak_probability.to_numpy(float) >= threshold
            values = np.where(gate[:, None], deep_values, inla_values)
            gate_scores.append(
                {
                    "parameter": float(threshold),
                    "deep_fraction": float(gate.mean()),
                    **metrics_for_values(part, values),
                }
            )
        key = str(int(horizon))
        soft_grid[key], gate_grid[key] = soft_scores, gate_scores
        selected["INLA-GNN-EXACT-SOFT-CRPS"][key] = min(
            soft_scores, key=lambda row: row["crps"]
        )["parameter"]
        selected["INLA-GNN-EXACT-SOFT-F1"][key] = max(
            soft_scores, key=lambda row: (row["f1"], -row["crps"])
        )["parameter"]
        selected["INLA-GNN-EXACT-GATE-CRPS"][key] = min(
            gate_scores, key=lambda row: row["crps"]
        )["parameter"]
        selected["INLA-GNN-EXACT-GATE-F1"][key] = max(
            gate_scores, key=lambda row: (row["f1"], -row["crps"])
        )["parameter"]
    return selected, {"soft": soft_grid, "gate": gate_grid}


def apply_policy(
    label: str,
    parameters: dict[str, float],
    inla_dir: Path,
    deep_dir: Path,
    output_dir: Path,
    draws: int,
):
    deep, deep_samples = load_deep(deep_dir, "test")
    samples = np.empty_like(deep_samples)
    predictions = deep.copy()
    predictions["model"] = label
    for horizon_index, horizon in enumerate(sorted(deep.horizon.unique())):
        mask = deep.horizon.eq(horizon)
        part = deep[mask].reset_index(drop=True)
        deep_values = deep_samples[:, :, horizon_index, :].reshape(-1, draws)
        inla_values = load_exact_horizon(inla_dir, part, int(horizon), draws)
        parameter = parameters[str(int(horizon))]
        if "SOFT" in label:
            deep_count = int(round(draws * parameter))
            values = np.concatenate(
                (deep_values[:, :deep_count], inla_values[:, : draws - deep_count]),
                axis=1,
            )
        else:
            gate = part.outbreak_probability.to_numpy(float) >= parameter
            values = np.where(gate[:, None], deep_values, inla_values)
        samples[:, :, horizon_index, :] = values.reshape(
            deep_samples.shape[0], deep_samples.shape[1], draws
        )
        predictions.loc[mask, "predicted_mean"] = values.mean(axis=1)
        predictions.loc[mask, "predicted_median"] = np.median(values, axis=1)
        for level, lo, hi in (
            (50, 0.25, 0.75), (80, 0.10, 0.90), (95, 0.025, 0.975)
        ):
            predictions.loc[mask, f"lower_{level}"] = np.quantile(values, lo, axis=1)
            predictions.loc[mask, f"upper_{level}"] = np.quantile(values, hi, axis=1)
        predictions.loc[mask, "outbreak_probability"] = (
            values > part.outbreak_threshold.to_numpy(float)[:, None]
        ).mean(axis=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    score_predictions(predictions, samples).to_csv(output_dir / "metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inla_validation_dir", type=Path)
    parser.add_argument("inla_test_dir", type=Path)
    parser.add_argument("deep_result_dir", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--draws", type=int, default=500)
    args = parser.parse_args()
    policies, grids = select_policies(
        args.inla_validation_dir, args.deep_result_dir, args.draws
    )
    for label, parameters in policies.items():
        apply_policy(
            label, parameters, args.inla_test_dir, args.deep_result_dir,
            args.output_root / label, args.draws,
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "selection.json").write_text(
        json.dumps(
            {
                "selection_split": "validation",
                "inla_distribution": "exact saved posterior-predictive draws",
                "policies": policies,
                "validation_grid": grids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
