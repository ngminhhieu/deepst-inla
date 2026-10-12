#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_h3.evaluation import _crps, outbreak_classification_metrics, score_predictions


KEYS = ["fcode", "issue_date", "target_date", "horizon"]


def align_inla(inla: pd.DataFrame, deep: pd.DataFrame) -> pd.DataFrame:
    indexed = inla.set_index(KEYS)
    keys = pd.MultiIndex.from_frame(deep[KEYS])
    aligned = indexed.reindex(keys).reset_index()
    if aligned["predicted_mean"].isna().any():
        raise ValueError("INLA and deep prediction rows do not align")
    return aligned


def reconstruct_inla_samples(
    inla: pd.DataFrame, draws: int, seed: int
) -> np.ndarray:
    lower = np.log1p(np.maximum(inla["lower_80"].to_numpy(float), 0))
    upper = np.log1p(np.maximum(inla["upper_80"].to_numpy(float), 0))
    sigma = np.clip((upper - lower) / (2 * 1.2815515655446004), 0.05, 3.0)
    mean = np.maximum(inla["predicted_mean"].to_numpy(float), 0)
    mu = np.log1p(mean) - 0.5 * np.square(sigma)
    latent = np.random.default_rng(seed).normal(
        mu[:, None], sigma[:, None], size=(len(inla), draws)
    )
    return np.maximum(np.expm1(latent), 0).astype(np.float32)


def load_deep(deep_dir: Path, split: str):
    prefix = "validation_" if split == "validation" else ""
    predictions = pd.read_csv(
        deep_dir / f"{prefix}predictions.csv", dtype={"fcode": str}
    )
    samples = np.load(deep_dir / f"{prefix}predictive_samples.npz")["samples"]
    return predictions, samples


def select_weights(inla_validation: Path, deep_dir: Path, draws: int, seed: int):
    inla = pd.read_csv(inla_validation, dtype={"fcode": str})
    deep, deep_samples = load_deep(deep_dir, "validation")
    inla = align_inla(inla, deep)
    candidates = np.linspace(0, 1, 11)
    grids, crps_weights, f1_weights = {}, {}, {}
    for horizon_index, horizon in enumerate(sorted(deep.horizon.unique())):
        selected = deep.horizon.eq(horizon)
        part = deep[selected].reset_index(drop=True)
        inla_part = inla[selected].reset_index(drop=True)
        deep_values = deep_samples[:, :, horizon_index, :].reshape(-1, draws)
        inla_values = reconstruct_inla_samples(
            inla_part, draws, seed + int(horizon)
        )
        observed = part.target_cases.to_numpy(float)
        population = part.population.to_numpy(float)
        observed_incidence = observed / population * 100_000
        scores = []
        for weight in candidates:
            deep_count = int(round(draws * weight))
            values = np.concatenate(
                (deep_values[:, :deep_count], inla_values[:, : draws - deep_count]),
                axis=1,
            )
            incidence = values / population[:, None] * 100_000
            probability = (
                values > part.outbreak_threshold.to_numpy(float)[:, None]
            ).mean(axis=1)
            classification = outbreak_classification_metrics(
                observed, part.outbreak_threshold.to_numpy(float), probability
            )
            scores.append(
                {
                    "deep_weight": float(weight),
                    "crps": float(_crps(observed_incidence, incidence).mean()),
                    "f1": float(classification["outbreak_f1"]),
                    "precision": float(classification["outbreak_precision"]),
                    "specificity": float(classification["outbreak_specificity"]),
                    "recall": float(classification["outbreak_recall"]),
                    "sensitivity": float(classification["outbreak_sensitivity"]),
                }
            )
        grids[str(int(horizon))] = scores
        crps_weights[str(int(horizon))] = min(scores, key=lambda row: row["crps"])[
            "deep_weight"
        ]
        f1_weights[str(int(horizon))] = max(
            scores, key=lambda row: (row["f1"], -row["crps"])
        )["deep_weight"]
    return crps_weights, f1_weights, grids


def apply_weights(
    inla_test: Path,
    deep_dir: Path,
    output_dir: Path,
    label: str,
    weights: dict[str, float],
    draws: int,
    seed: int,
):
    inla = pd.read_csv(inla_test, dtype={"fcode": str})
    deep, deep_samples = load_deep(deep_dir, "test")
    inla = align_inla(inla, deep)
    horizons = sorted(deep.horizon.unique())
    samples = np.empty_like(deep_samples)
    predictions = deep.copy()
    predictions["model"] = label
    for horizon_index, horizon in enumerate(horizons):
        selected = deep.horizon.eq(horizon)
        part = deep[selected].reset_index(drop=True)
        inla_part = inla[selected].reset_index(drop=True)
        deep_values = deep_samples[:, :, horizon_index, :].reshape(-1, draws)
        inla_values = reconstruct_inla_samples(
            inla_part, draws, seed + 100 + int(horizon)
        )
        deep_count = int(round(draws * weights[str(int(horizon))]))
        values = np.concatenate(
            (deep_values[:, :deep_count], inla_values[:, : draws - deep_count]),
            axis=1,
        )
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
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    score_predictions(predictions, samples).to_csv(
        output_dir / "metrics.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inla_validation_predictions", type=Path)
    parser.add_argument("inla_test_predictions", type=Path)
    parser.add_argument("deep_result_dir", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    crps_weights, f1_weights, grid = select_weights(
        args.inla_validation_predictions,
        args.deep_result_dir,
        args.draws,
        args.seed,
    )
    policies = {"INLA-GNN-CRPS": crps_weights, "INLA-GNN-F1": f1_weights}
    for label, weights in policies.items():
        apply_weights(
            args.inla_test_predictions,
            args.deep_result_dir,
            args.output_root / label,
            label,
            weights,
            args.draws,
            args.seed,
        )
    (args.output_root / "selection.json").write_text(
        json.dumps(
            {
                "selection_split": "validation",
                "inla_distribution": "lognormal reconstruction from mean and 80% interval",
                "policies": policies,
                "validation_grid": grid,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
