#!/usr/bin/env python3
"""Validation-selected ensemble targeting count, calibration, and alert metrics."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_h3.evaluation import _crps, outbreak_classification_metrics, score_predictions
from ensemble_inla_deep import load_deep
from ensemble_model_predictions import aligned_samples
from exact_inla_deep_ensemble import load_exact_horizon


IDENTITY = ["fcode", "issue_date", "target_date", "horizon"]


def compositions(total: int, members: int):
    if members == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, members - 1):
            yield (first, *rest)


def mix_arrays(arrays: list[np.ndarray], weights: tuple[float, ...], draws: int) -> np.ndarray:
    counts = [int(round(draws * weight)) for weight in weights]
    counts[0] += draws - sum(counts)
    pieces = [array[:, :count] for array, count in zip(arrays, counts) if count]
    return np.concatenate(pieces, axis=1)


def distribution_metrics(part: pd.DataFrame, values: np.ndarray) -> dict[str, float]:
    population = part["population"].to_numpy(float)
    observed = part["target_cases"].to_numpy(float) / population * 100_000
    incidence = values / population[:, None] * 100_000
    error = incidence.mean(axis=1) - observed
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "crps": float(_crps(observed, incidence).mean()),
    }


def choose_distribution(
    part: pd.DataFrame,
    deep: np.ndarray,
    peak: np.ndarray,
    hybrid: np.ndarray,
    peak_risk: np.ndarray,
    draws: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    arrays = [hybrid[:, :draws], deep[:, :draws], peak[:, :draws]]
    for counts in compositions(10, 3):
        weights = tuple(value / 10 for value in counts)
        values = mix_arrays(arrays, weights, draws)
        candidates.append(
            {"kind": "soft", "weights": list(weights), **distribution_metrics(part, values)}
        )
    for expert_name, expert in (("deep", deep), ("warning", peak)):
        for gate_threshold in np.arange(0.10, 0.71, 0.10):
            gate = peak_risk >= gate_threshold
            for expert_weight in np.arange(0.10, 0.81, 0.10):
                mixed = mix_arrays(
                    [hybrid[:, :draws], expert[:, :draws]],
                    (1.0 - float(expert_weight), float(expert_weight)),
                    draws,
                )
                values = np.where(gate[:, None], mixed, hybrid[:, :draws])
                candidates.append(
                    {
                        "kind": "risk_gate",
                        "expert": expert_name,
                        "gate_threshold": float(gate_threshold),
                        "expert_weight": float(expert_weight),
                        "gate_fraction": float(gate.mean()),
                        **distribution_metrics(part, values),
                    }
                )
    for metric in ("mae", "rmse", "crps"):
        values = np.array([float(row[metric]) for row in candidates])
        lower, upper = float(values.min()), float(values.max())
        scale = max(upper - lower, 1e-12)
        for row in candidates:
            row[f"normalised_{metric}_loss"] = (float(row[metric]) - lower) / scale
    for row in candidates:
        row["distribution_objective"] = float(
            np.mean([row["normalised_mae_loss"], row["normalised_rmse_loss"], row["normalised_crps_loss"]])
        )
    winner = min(candidates, key=lambda row: row["distribution_objective"])
    return winner, candidates


def apply_distribution_policy(
    policy: dict[str, object],
    deep: np.ndarray,
    peak: np.ndarray,
    hybrid: np.ndarray,
    peak_risk: np.ndarray,
    draws: int,
) -> np.ndarray:
    if policy["kind"] == "soft":
        return mix_arrays(
            [hybrid, deep, peak], tuple(float(x) for x in policy["weights"]), draws
        )
    expert = deep if policy["expert"] == "deep" else peak
    weight = float(policy["expert_weight"])
    mixed = mix_arrays([hybrid, expert], (1.0 - weight, weight), draws)
    gate = peak_risk >= float(policy["gate_threshold"])
    return np.where(gate[:, None], mixed, hybrid)


def choose_alert_head(
    part: pd.DataFrame,
    probabilities: list[np.ndarray],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    observed = part["target_cases"].to_numpy(float)
    outbreak_threshold = part["outbreak_threshold"].to_numpy(float)
    observed_binary = observed > outbreak_threshold
    candidates: list[dict[str, object]] = []
    for counts in compositions(5, len(probabilities)):
        weights = tuple(value / 5 for value in counts)
        raw_probability = sum(weight * member for weight, member in zip(weights, probabilities))
        intercept, slope = fit_platt(raw_probability, observed_binary.astype(float))
        probability = apply_platt(raw_probability, intercept, slope)
        brier = float(np.square(probability - observed_binary).mean())
        for threshold in np.arange(0.15, 0.751, 0.02):
            metrics = outbreak_classification_metrics(
                observed, outbreak_threshold, probability, decision_threshold=float(threshold)
            )
            precision = float(metrics["outbreak_precision"])
            specificity = float(metrics["outbreak_specificity"])
            sensitivity = float(metrics["outbreak_sensitivity"])
            f1 = float(metrics["outbreak_f1"])
            if precision < 0.50 or specificity < 0.90:
                continue
            objective = float(
                np.prod(
                    np.clip(
                        [precision, specificity, sensitivity, f1, 1.0 - brier],
                        1e-9,
                        1.0,
                    )
                ) ** (1.0 / 5.0)
            )
            candidates.append(
                {
                    "weights": list(weights),
                    "platt_intercept": intercept,
                    "platt_slope": slope,
                    "decision_threshold": float(threshold),
                    "brier": brier,
                    "precision": precision,
                    "specificity": specificity,
                    "sensitivity": sensitivity,
                    "f1": f1,
                    "alert_objective": objective,
                }
            )
    winner = max(candidates, key=lambda row: row["alert_objective"])
    return winner, candidates


def apply_platt(probability: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    clipped = np.clip(probability, 1e-5, 1.0 - 1e-5)
    logit = np.log(clipped / (1.0 - clipped))
    linear = np.clip(intercept + slope * logit, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-linear))


def fit_platt(probability: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-5, 1.0 - 1e-5)
    logit = np.log(clipped / (1.0 - clipped))
    design = np.column_stack((np.ones(len(logit)), logit))
    coefficients = np.array([0.0, 1.0], dtype=float)
    penalty = np.diag([0.0, 1e-3])
    for _ in range(50):
        linear = np.clip(design @ coefficients, -30.0, 30.0)
        fitted = 1.0 / (1.0 + np.exp(-linear))
        weights = np.clip(fitted * (1.0 - fitted), 1e-6, None)
        gradient = design.T @ (fitted - labels) + penalty @ coefficients
        hessian = design.T @ (weights[:, None] * design) + penalty
        update = np.linalg.solve(hessian, gradient)
        coefficients -= update
        if float(np.max(np.abs(update))) < 1e-8:
            break
    return float(coefficients[0]), float(coefficients[1])


def aligned_column(template: pd.DataFrame, source: pd.DataFrame, column: str) -> np.ndarray:
    indexed = source.set_index(IDENTITY)
    keys = pd.MultiIndex.from_frame(template[IDENTITY])
    values = indexed.reindex(keys)[column]
    if values.isna().any():
        raise ValueError(f"Could not align {column}")
    return values.to_numpy(float)


def run_split(
    split: str,
    deep_dir: Path,
    peak_dir: Path,
    hybrid_dir: Path,
    policies: dict[int, dict[str, object]] | None,
    draws: int,
    selection_draws: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[int, dict[str, object]], dict[str, object]]:
    deep_frame, deep_cube = load_deep(deep_dir, split)
    peak_frame, peak_cube_raw = load_deep(peak_dir, split)
    peak_cube = aligned_samples(deep_frame, peak_frame, peak_cube_raw)
    horizons = sorted(int(value) for value in deep_frame["horizon"].unique())
    output_cube = np.empty_like(deep_cube)
    predictions = deep_frame.copy()
    predictions["model"] = "MULTIOBJECTIVE-ENSEMBLE"
    selected: dict[int, dict[str, object]] = {} if policies is None else policies
    diagnostics: dict[str, object] = {}
    for horizon_index, horizon in enumerate(horizons):
        mask = deep_frame["horizon"].eq(horizon)
        part = deep_frame.loc[mask].reset_index(drop=True)
        deep_values = deep_cube[:, :, horizon_index, :].reshape(-1, draws)
        peak_values = peak_cube[:, :, horizon_index, :].reshape(-1, draws)
        hybrid_values = load_exact_horizon(hybrid_dir, part, horizon, draws)
        peak_part = peak_frame.loc[peak_frame["horizon"].eq(horizon)].reset_index(drop=True)
        peak_risk = aligned_column(part, peak_part, "outbreak_probability")
        if policies is None:
            distribution_policy, distribution_grid = choose_distribution(
                part,
                deep_values,
                peak_values,
                hybrid_values,
                peak_risk,
                selection_draws,
            )
        else:
            distribution_policy = policies[horizon]["distribution"]
            distribution_grid = []
        values = apply_distribution_policy(
            distribution_policy,
            deep_values,
            peak_values,
            hybrid_values,
            peak_risk,
            draws,
        )
        distribution_probability = (
            values > part["outbreak_threshold"].to_numpy(float)[:, None]
        ).mean(axis=1)
        deep_probability = (
            deep_values > part["outbreak_threshold"].to_numpy(float)[:, None]
        ).mean(axis=1)
        peak_probability = (
            peak_values > part["outbreak_threshold"].to_numpy(float)[:, None]
        ).mean(axis=1)
        peak_auxiliary = aligned_column(part, peak_part, "outbreak_probability_auxiliary")
        probability_members = [
            distribution_probability,
            deep_probability,
            peak_probability,
            peak_auxiliary,
        ]
        if policies is None:
            alert_policy, alert_grid = choose_alert_head(part, probability_members)
            selected[horizon] = {"distribution": distribution_policy, "alert": alert_policy}
        else:
            alert_policy = policies[horizon]["alert"]
            alert_grid = []
        raw_alert_probability = sum(
            float(weight) * member
            for weight, member in zip(alert_policy["weights"], probability_members)
        )
        alert_probability = apply_platt(
            raw_alert_probability,
            float(alert_policy["platt_intercept"]),
            float(alert_policy["platt_slope"]),
        )
        output_cube[:, :, horizon_index, :] = values.reshape(
            deep_cube.shape[0], deep_cube.shape[1], draws
        )
        predictions.loc[mask, "predicted_mean"] = values.mean(axis=1)
        predictions.loc[mask, "predicted_median"] = np.median(values, axis=1)
        for level, lower, upper in ((50, .25, .75), (80, .10, .90), (95, .025, .975)):
            predictions.loc[mask, f"lower_{level}"] = np.quantile(values, lower, axis=1)
            predictions.loc[mask, f"upper_{level}"] = np.quantile(values, upper, axis=1)
        predictions.loc[mask, "outbreak_probability"] = alert_probability
        predictions.loc[mask, "outbreak_decision_threshold"] = float(
            alert_policy["decision_threshold"]
        )
        diagnostics[str(horizon)] = {
            "distribution_candidates": len(distribution_grid),
            "alert_candidates": len(alert_grid),
            "selected": selected[horizon],
        }
    return predictions, output_cube, selected, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", type=Path, required=True)
    parser.add_argument("--peak", type=Path, required=True)
    parser.add_argument("--hybrid-validation", type=Path, required=True)
    parser.add_argument("--hybrid-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--selection-draws", type=int, default=100)
    parser.add_argument(
        "--selection-split-label",
        default="validation 2019-2020",
        help="Human-readable validation period recorded in selection.json.",
    )
    args = parser.parse_args()
    validation, validation_samples, policies, diagnostics = run_split(
        "validation",
        args.deep,
        args.peak,
        args.hybrid_validation,
        None,
        args.draws,
        args.selection_draws,
    )
    test, test_samples, _, _ = run_split(
        "test",
        args.deep,
        args.peak,
        args.hybrid_test,
        policies,
        args.draws,
        args.selection_draws,
    )
    thresholds = {
        horizon: float(policy["alert"]["decision_threshold"])
        for horizon, policy in policies.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    validation.to_csv(args.output / "validation_predictions.csv", index=False)
    test.to_csv(args.output / "predictions.csv", index=False)
    np.savez_compressed(
        args.output / "validation_predictive_samples.npz", samples=validation_samples
    )
    np.savez_compressed(args.output / "predictive_samples.npz", samples=test_samples)
    score_predictions(
        validation,
        validation_samples,
        decision_threshold=thresholds,
        split="validation",
    ).to_csv(args.output / "validation_metrics.csv", index=False)
    score_predictions(test, test_samples, decision_threshold=thresholds).to_csv(
        args.output / "metrics.csv", index=False
    )
    (args.output / "selection.json").write_text(
        json.dumps(
            {
                "selection_split": args.selection_split_label,
                "distribution_objective": "equal normalised MAE, RMSE, and CRPS loss",
                "alert_objective": "geometric mean of precision, specificity, sensitivity, F1, and one minus Brier",
                "alert_probability_members": [
                    "selected distribution exceedance probability",
                    "TCN-GNN exceedance probability",
                    "focal-BCE warning-expert exceedance probability",
                    "trained focal-BCE auxiliary outbreak head",
                ],
                "policies": policies,
                "diagnostics": diagnostics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
