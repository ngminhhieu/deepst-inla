#!/usr/bin/env python3
"""Tune a partially blended peak gate on validation and apply it once to test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_h3.evaluation import _crps, score_predictions
from ensemble_inla_deep import load_deep
from exact_inla_deep_ensemble import load_exact_horizon


PRIMARY_LOSSES = ("mae", "rmse", "crps", "outbreak_brier")
PRIMARY_GAINS = ("outbreak_precision", "outbreak_sensitivity", "outbreak_f1")


def candidate_values(
    base: np.ndarray,
    peak: np.ndarray,
    gate: np.ndarray,
    peak_weight: float,
) -> np.ndarray:
    peak_count = int(round(base.shape[1] * peak_weight))
    if peak_count == 0:
        mixed = base
    elif peak_count == base.shape[1]:
        mixed = peak
    else:
        mixed = np.concatenate(
            (peak[:, :peak_count], base[:, : base.shape[1] - peak_count]), axis=1
        )
    return np.where(gate[:, None], mixed, base)


def distribution_metrics(
    part: pd.DataFrame,
    mean: np.ndarray,
    probability: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    threshold = part["outbreak_threshold"].to_numpy(float)
    observed_cases = part["target_cases"].to_numpy(float)
    population = part["population"].to_numpy(float)
    observed_outbreak = observed_cases > threshold
    error = (mean - observed_cases) / population * 100_000
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "outbreak_brier": float(np.square(probability - observed_outbreak).mean()),
    }, observed_outbreak


def classification_metrics(
    probability: np.ndarray,
    observed_outbreak: np.ndarray,
    decision_threshold: float,
) -> dict[str, float]:
    alert = probability >= decision_threshold
    tp = int(np.sum(alert & observed_outbreak))
    fp = int(np.sum(alert & ~observed_outbreak))
    fn = int(np.sum(~alert & observed_outbreak))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    tn = int(np.sum((~alert) & (~observed_outbreak)))
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "outbreak_precision": precision,
        "outbreak_specificity": specificity,
        "outbreak_recall": recall,
        "outbreak_sensitivity": recall,
        "outbreak_f1": f1,
    }


def dominates_benchmark(row: dict[str, float], benchmark: pd.Series) -> bool:
    return all(row[key] < float(benchmark[key]) for key in PRIMARY_LOSSES if key != "crps") and all(
        row[key] > float(benchmark[key]) for key in PRIMARY_GAINS
    )


def select_policies(
    base_dir: Path,
    peak_dir: Path,
    benchmark_metrics_path: Path,
    draws: int,
) -> tuple[dict[int, dict[str, float]], dict[str, list[dict[str, float]]]]:
    deep, deep_samples = load_deep(peak_dir, "validation")
    benchmarks = pd.read_csv(benchmark_metrics_path).set_index("horizon")
    selected: dict[int, dict[str, float]] = {}
    grids: dict[str, list[dict[str, float]]] = {}
    horizons = sorted(int(value) for value in deep["horizon"].unique())
    for horizon_index, horizon in enumerate(horizons):
        part = deep[deep["horizon"].eq(horizon)].reset_index(drop=True)
        peak = deep_samples[:, :, horizon_index, :].reshape(-1, draws).astype(np.float32)
        base = load_exact_horizon(base_dir, part, horizon, draws)
        signal = part["outbreak_probability"].to_numpy(float)
        benchmark = benchmarks.loc[horizon]
        feasible = []
        threshold = part["outbreak_threshold"].to_numpy(float)
        base_mean = base.mean(axis=1)
        base_probability = (base > threshold[:, None]).mean(axis=1)
        mixtures = {}
        for peak_weight in np.arange(0.1, 1.001, 0.1):
            peak_count = int(round(draws * peak_weight))
            base_count = draws - peak_count
            mixtures[float(peak_weight)] = (
                (
                    peak[:, :peak_count].sum(axis=1)
                    + base[:, :base_count].sum(axis=1)
                )
                / draws,
                (
                    (peak[:, :peak_count] > threshold[:, None]).sum(axis=1)
                    + (base[:, :base_count] > threshold[:, None]).sum(axis=1)
                )
                / draws,
            )
        for gate_threshold in np.arange(0.05, 0.951, 0.05):
            gate = signal >= gate_threshold
            for peak_weight, (mixed_mean, mixed_probability) in mixtures.items():
                mean = np.where(gate, mixed_mean, base_mean)
                probability = np.where(gate, mixed_probability, base_probability)
                distribution, observed_outbreak = distribution_metrics(
                    part, mean, probability
                )
                for decision_threshold in np.arange(0.50, 0.751, 0.025):
                    row = {
                        "gate_threshold": float(gate_threshold),
                        "peak_weight": float(peak_weight),
                        "decision_threshold": float(decision_threshold),
                        "gate_fraction": float(gate.mean()),
                        **distribution,
                        **classification_metrics(
                            probability, observed_outbreak, float(decision_threshold)
                        ),
                    }
                    if dominates_benchmark(row, benchmark):
                        feasible.append(row)
        if not feasible:
            raise RuntimeError(
                f"No H{horizon} policy dominates INLA on validation primary metrics"
            )

        # CRPS is expensive. Evaluate it in exact draws for candidates ordered
        # by the optimisation objective and retain the first one that also beats INLA.
        feasible.sort(key=lambda row: (-row["outbreak_f1"], row["rmse"], row["mae"]))
        evaluated = []
        for row in feasible:
            gate = signal >= row["gate_threshold"]
            values = candidate_values(base, peak, gate, row["peak_weight"])
            incidence = values / part["population"].to_numpy(float)[:, None] * 100_000
            observed = (
                part["target_cases"].to_numpy(float)
                / part["population"].to_numpy(float)
                * 100_000
            )
            row = dict(row)
            row["crps"] = float(_crps(observed, incidence).mean())
            evaluated.append(row)
            if row["crps"] < float(benchmark["crps"]):
                selected[horizon] = row
                break
        if horizon not in selected:
            raise RuntimeError(
                f"No H{horizon} policy also dominates INLA validation CRPS"
            )
        grids[str(horizon)] = evaluated
    return selected, grids


def apply_policies(
    policies: dict[int, dict[str, float]],
    base_dir: Path,
    peak_dir: Path,
    output_dir: Path,
    split: str,
    draws: int,
) -> None:
    deep, deep_samples = load_deep(peak_dir, split)
    samples = np.empty_like(deep_samples)
    predictions = deep.copy()
    predictions["model"] = "PROPOSAL-TUNED"
    horizons = sorted(int(value) for value in deep["horizon"].unique())
    decision_thresholds = {}
    for horizon_index, horizon in enumerate(horizons):
        mask = deep["horizon"].eq(horizon)
        part = deep[mask].reset_index(drop=True)
        peak = deep_samples[:, :, horizon_index, :].reshape(-1, draws)
        base = load_exact_horizon(base_dir, part, horizon, draws)
        policy = policies[horizon]
        gate = part["outbreak_probability"].to_numpy(float) >= policy["gate_threshold"]
        values = candidate_values(base, peak, gate, policy["peak_weight"])
        samples[:, :, horizon_index, :] = values.reshape(
            deep_samples.shape[0], deep_samples.shape[1], draws
        )
        predictions.loc[mask, "predicted_mean"] = values.mean(axis=1)
        predictions.loc[mask, "predicted_median"] = np.median(values, axis=1)
        for level, lo, hi in ((50, .25, .75), (80, .10, .90), (95, .025, .975)):
            predictions.loc[mask, f"lower_{level}"] = np.quantile(values, lo, axis=1)
            predictions.loc[mask, f"upper_{level}"] = np.quantile(values, hi, axis=1)
        predictions.loc[mask, "outbreak_probability"] = (
            values > part["outbreak_threshold"].to_numpy(float)[:, None]
        ).mean(axis=1)
        predictions.loc[mask, "outbreak_decision_threshold"] = policy[
            "decision_threshold"
        ]
        predictions.loc[mask, "peak_gate_active"] = gate
        predictions.loc[mask, "peak_gate_threshold"] = policy["gate_threshold"]
        predictions.loc[mask, "peak_weight"] = policy["peak_weight"]
        decision_thresholds[horizon] = policy["decision_threshold"]

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "validation_" if split == "validation" else ""
    predictions.to_csv(output_dir / f"{prefix}predictions.csv", index=False)
    np.savez_compressed(output_dir / f"{prefix}predictive_samples.npz", samples=samples)
    score_predictions(
        predictions,
        samples,
        decision_threshold=decision_thresholds,
        split=split,
    ).to_csv(output_dir / f"{prefix}metrics.csv", index=False)


def select_global_decision_threshold(
    validation_predictions_path: Path,
    benchmark_metrics_path: Path,
) -> tuple[float, list[dict[str, object]]]:
    """Choose the most conservative shared alert cutoff retaining dominance."""
    predictions = pd.read_csv(validation_predictions_path)
    benchmarks = pd.read_csv(benchmark_metrics_path).set_index("horizon")
    feasible = []
    for threshold in np.arange(0.50, 0.751, 0.001):
        horizon_rows = []
        dominates_all = True
        for horizon in sorted(int(value) for value in predictions["horizon"].unique()):
            part = predictions[predictions["horizon"].eq(horizon)]
            observed = part["target_cases"] > part["outbreak_threshold"]
            probability = part["outbreak_probability"].to_numpy(float)
            metrics = classification_metrics(
                probability, observed.to_numpy(bool), float(threshold)
            )
            benchmark = benchmarks.loc[horizon]
            dominates = all(
                metrics[key] > float(benchmark[key]) for key in PRIMARY_GAINS
            )
            dominates_all &= dominates
            horizon_rows.append(
                {"horizon": horizon, "dominates": dominates, **metrics}
            )
        if dominates_all:
            feasible.append(
                {"decision_threshold": float(threshold), "horizons": horizon_rows}
            )
    if not feasible:
        raise RuntimeError(
            "No shared validation alert cutoff dominates INLA at every horizon"
        )
    # Highest feasible threshold is the most conservative operating point and
    # protects precision while preserving strict sensitivity and F1 improvements.
    selected = max(row["decision_threshold"] for row in feasible)
    return selected, feasible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_base_dir", type=Path)
    parser.add_argument("test_base_dir", type=Path)
    parser.add_argument("peak_dir", type=Path)
    parser.add_argument("benchmark_validation_metrics", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--draws", type=int, default=500)
    args = parser.parse_args()
    policies, grids = select_policies(
        args.validation_base_dir,
        args.peak_dir,
        args.benchmark_validation_metrics,
        args.draws,
    )
    apply_policies(
        policies, args.validation_base_dir, args.peak_dir, args.output_dir,
        "validation", args.draws,
    )
    global_decision_threshold, decision_grid = select_global_decision_threshold(
        args.output_dir / "validation_predictions.csv",
        args.benchmark_validation_metrics,
    )
    for policy in policies.values():
        policy["decision_threshold"] = global_decision_threshold
    # Rewrite validation metrics/predictions with the locked shared threshold.
    apply_policies(
        policies, args.validation_base_dir, args.peak_dir, args.output_dir,
        "validation", args.draws,
    )
    final_validation = pd.read_csv(
        args.output_dir / "validation_metrics.csv"
    ).set_index("horizon")
    for horizon, policy in policies.items():
        for key in (
            "mae", "rmse", "crps", "outbreak_brier", "outbreak_precision",
            "outbreak_specificity", "outbreak_recall", "outbreak_sensitivity", "outbreak_f1",
        ):
            policy[key] = float(final_validation.loc[horizon, key])
    apply_policies(
        policies, args.test_base_dir, args.peak_dir, args.output_dir,
        "test", args.draws,
    )
    (args.output_dir / "selection.json").write_text(
        json.dumps(
            {
                "selection_split": "validation",
                "objective": "select partial-gate distributions by validation F1 subject to strict improvement over INLA-TS in MAE, RMSE, CRPS, Brier, precision, sensitivity, and F1; then choose the highest shared H1-H3 alert cutoff retaining strict precision, sensitivity, and F1 dominance",
                "policies": {str(key): value for key, value in policies.items()},
                "global_decision_threshold": global_decision_threshold,
                "feasible_global_decision_grid": decision_grid,
                "evaluated_crps_frontier": grids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({str(key): value for key, value in policies.items()}, indent=2))


if __name__ == "__main__":
    main()
