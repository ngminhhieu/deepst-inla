#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_h3.evaluation import _crps, outbreak_classification_metrics, score_predictions
from ensemble_inla_deep import (
    align_inla,
    load_deep,
    reconstruct_inla_samples,
)


def select_gates(inla_path: Path, deep_dir: Path, draws: int, seed: int):
    inla = pd.read_csv(inla_path, dtype={"fcode": str})
    deep, deep_samples = load_deep(deep_dir, "validation")
    inla = align_inla(inla, deep)
    candidates = np.concatenate(([-0.01], np.arange(0.05, 0.55, 0.05), [1.01]))
    grids, crps_thresholds, f1_thresholds = {}, {}, {}
    for horizon_index, horizon in enumerate(sorted(deep.horizon.unique())):
        selected = deep.horizon.eq(horizon)
        part = deep[selected].reset_index(drop=True)
        deep_values = deep_samples[:, :, horizon_index, :].reshape(-1, draws)
        inla_values = reconstruct_inla_samples(
            inla[selected].reset_index(drop=True), draws, seed + int(horizon)
        )
        observed = part.target_cases.to_numpy(float)
        population = part.population.to_numpy(float)
        observed_incidence = observed / population * 100_000
        scores = []
        for threshold in candidates:
            gate = part.outbreak_probability.to_numpy(float) >= threshold
            values = np.where(gate[:, None], deep_values, inla_values)
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
                    "deep_fraction": float(gate.mean()),
                    "crps": float(_crps(observed_incidence, incidence).mean()),
                    "f1": float(classification["outbreak_f1"]),
                    "precision": float(classification["outbreak_precision"]),
                    "specificity": float(classification["outbreak_specificity"]),
                    "recall": float(classification["outbreak_recall"]),
                    "sensitivity": float(classification["outbreak_sensitivity"]),
                }
            )
        grids[str(int(horizon))] = scores
        crps_thresholds[str(int(horizon))] = min(
            scores, key=lambda row: row["crps"]
        )["threshold"]
        f1_thresholds[str(int(horizon))] = max(
            scores, key=lambda row: (row["f1"], -row["crps"])
        )["threshold"]
    return crps_thresholds, f1_thresholds, grids


def apply_gate(
    inla_path: Path,
    deep_dir: Path,
    output_dir: Path,
    label: str,
    thresholds: dict[str, float],
    draws: int,
    seed: int,
):
    inla = pd.read_csv(inla_path, dtype={"fcode": str})
    deep, deep_samples = load_deep(deep_dir, "test")
    inla = align_inla(inla, deep)
    samples = np.empty_like(deep_samples)
    predictions = deep.copy()
    predictions["model"] = label
    fractions = {}
    for horizon_index, horizon in enumerate(sorted(deep.horizon.unique())):
        selected = deep.horizon.eq(horizon)
        part = deep[selected].reset_index(drop=True)
        deep_values = deep_samples[:, :, horizon_index, :].reshape(-1, draws)
        inla_values = reconstruct_inla_samples(
            inla[selected].reset_index(drop=True), draws, seed + 100 + int(horizon)
        )
        gate = (
            part.outbreak_probability.to_numpy(float)
            >= thresholds[str(int(horizon))]
        )
        values = np.where(gate[:, None], deep_values, inla_values)
        samples[:, :, horizon_index, :] = values.reshape(
            deep_samples.shape[0], deep_samples.shape[1], draws
        )
        predictions.loc[selected, "predicted_mean"] = values.mean(axis=1)
        predictions.loc[selected, "predicted_median"] = np.median(values, axis=1)
        for level, lo, hi in (
            (50, 0.25, 0.75), (80, 0.10, 0.90), (95, 0.025, 0.975)
        ):
            predictions.loc[selected, f"lower_{level}"] = np.quantile(
                values, lo, axis=1
            )
            predictions.loc[selected, f"upper_{level}"] = np.quantile(
                values, hi, axis=1
            )
        predictions.loc[selected, "outbreak_probability"] = (
            values
            > part.outbreak_threshold.to_numpy(float)[:, None]
        ).mean(axis=1)
        fractions[str(int(horizon))] = float(gate.mean())
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    score_predictions(predictions, samples).to_csv(
        output_dir / "metrics.csv", index=False
    )
    return fractions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inla_validation_predictions", type=Path)
    parser.add_argument("inla_test_predictions", type=Path)
    parser.add_argument("deep_result_dir", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    crps_thresholds, f1_thresholds, grid = select_gates(
        args.inla_validation_predictions,
        args.deep_result_dir,
        args.draws,
        args.seed,
    )
    policies = {
        "INLA-GNN-GATE-CRPS": crps_thresholds,
        "INLA-GNN-GATE-F1": f1_thresholds,
    }
    fractions = {}
    for label, thresholds in policies.items():
        fractions[label] = apply_gate(
            args.inla_test_predictions,
            args.deep_result_dir,
            args.output_root / label,
            label,
            thresholds,
            args.draws,
            args.seed,
        )
    (args.output_root / "selection.json").write_text(
        json.dumps(
            {
                "selection_split": "validation",
                "gate_rule": "use deep when deep outbreak probability >= threshold",
                "inla_distribution": "lognormal reconstruction from mean and 80% interval",
                "policies": policies,
                "test_deep_fractions": fractions,
                "validation_grid": grid,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
