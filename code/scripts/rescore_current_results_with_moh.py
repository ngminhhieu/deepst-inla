#!/usr/bin/env python3
"""Re-score retained forecast distributions against the audited MOH thresholds.

This script does not refit count models.  It replaces every legacy outbreak
threshold in retained prediction artifacts, recomputes exceedance probabilities
from the retained predictive draws, selects only the decision cutoffs on the
2019--2020 validation split, and writes corrected 2021--2025 result tables.

The separately supplied 549-commune ensemble contains quantiles rather than
draws.  Its CDF and CRPS are therefore reconstructed deterministically by
linear interpolation of the supplied quantile function and are labelled as
approximations in the provenance output.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = (
    ROOT
    / "artifacts/recovered_from_trash/results/"
    "model_runs_train2010_2018_valid2019_2020_test2021_2025_external_thresholds"
)
METRIC_COLUMNS = (
    "mae",
    "rmse",
    "crps",
    "outbreak_brier",
    "outbreak_precision",
    "outbreak_specificity",
    "outbreak_sensitivity",
    "outbreak_f1",
)
LOSS_COLUMNS = {"mae", "rmse", "crps", "outbreak_brier"}


def load_moh(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"fcode": str})
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    frame["epi_threshold"] = pd.to_numeric(frame["epi_threshold"], errors="coerce")
    frame = frame.dropna(subset=["epi_threshold"])
    if frame.duplicated(["fcode", "target_date"]).any():
        raise ValueError("MOH table has duplicate commune-month rows")
    return frame[["fcode", "target_date", "epi_threshold"]]


def attach_moh(frame: pd.DataFrame, moh: pd.DataFrame) -> pd.DataFrame:
    output = frame.drop(columns=["outbreak_threshold"], errors="ignore").copy()
    output["fcode"] = output["fcode"].astype(str)
    output["target_date"] = pd.to_datetime(output["target_date"])
    output = output.merge(
        moh, on=["fcode", "target_date"], how="left", validate="many_to_one"
    ).rename(columns={"epi_threshold": "outbreak_threshold"})
    if output["outbreak_threshold"].isna().any():
        missing = output.loc[
            output["outbreak_threshold"].isna(), ["fcode", "target_date"]
        ].head()
        raise ValueError(f"MOH threshold missing for forecast rows:\n{missing}")
    return output


def load_npz_split(directory: Path, split: str) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    prefix = "validation_" if split == "validation" else ""
    frame = pd.read_csv(directory / f"{prefix}predictions.csv", dtype={"fcode": str})
    cube = np.load(directory / f"{prefix}predictive_samples.npz")["samples"]
    horizons = sorted(frame["horizon"].astype(int).unique())
    values = {
        horizon: cube[:, :, index, :].reshape(-1, cube.shape[-1])
        for index, horizon in enumerate(horizons)
    }
    for horizon, matrix in values.items():
        if len(matrix) != frame["horizon"].eq(horizon).sum():
            raise ValueError(f"H{horizon} prediction/sample mismatch in {directory}")
    return frame, values


def load_binary_split(directory: Path) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    frame = pd.read_csv(directory / "predictions.csv", dtype={"fcode": str})
    values: dict[int, np.ndarray] = {}
    for horizon in sorted(frame["horizon"].astype(int).unique()):
        shape = pd.read_csv(directory / f"samples_H{horizon}_shape.csv")
        rows, draws = int(shape.loc[0, "rows"]), int(shape.loc[0, "draws"])
        matrix = np.fromfile(directory / f"samples_H{horizon}.bin", dtype=np.float64)
        # R writes matrices in column-major order.
        values[horizon] = matrix.reshape(rows, draws, order="F")
        if rows != frame["horizon"].eq(horizon).sum():
            raise ValueError(f"H{horizon} prediction/sample mismatch in {directory}")
    return frame, values


def load_lognormal_validation(
    directory: Path, draws: int = 500
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Reconstruct validation draws when only lognormal parameters were retained."""
    frame = pd.read_csv(directory / "validation_predictions.csv", dtype={"fcode": str})
    grid = (np.arange(draws) + 0.5) / draws
    z = np.asarray([NormalDist().inv_cdf(float(value)) for value in grid])
    values = {}
    for horizon in sorted(frame["horizon"].astype(int).unique()):
        part = frame[frame["horizon"].eq(horizon)]
        location = part["log_location"].to_numpy(float)
        scale = part["log_scale"].to_numpy(float)
        values[int(horizon)] = np.maximum(
            np.expm1(location[:, None] + scale[:, None] * z[None, :]), 0
        ).astype(np.float32)
    return frame, values


def classification(labels: np.ndarray, probability: np.ndarray, cutoff: float) -> dict[str, float | int]:
    predicted = probability >= cutoff
    tp = int(np.sum(labels & predicted))
    fp = int(np.sum(~labels & predicted))
    fn = int(np.sum(labels & ~predicted))
    tn = int(np.sum(~labels & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "outbreak_decision_threshold": float(cutoff),
        "outbreak_observed_positive": int(labels.sum()),
        "outbreak_predicted_positive": int(predicted.sum()),
        "outbreak_tp": tp,
        "outbreak_fp": fp,
        "outbreak_fn": fn,
        "outbreak_tn": tn,
        "outbreak_accuracy": (tp + tn) / len(labels),
        "outbreak_precision": precision,
        "outbreak_specificity": specificity,
        "outbreak_recall": sensitivity,
        "outbreak_sensitivity": sensitivity,
        "outbreak_true_negative_rate": specificity,
        "outbreak_balanced_accuracy": (sensitivity + specificity) / 2,
        "outbreak_f1": f1,
    }


def select_cutoff(
    labels: np.ndarray,
    probability: np.ndarray,
    policy: str,
) -> float:
    if policy == "fixed_0.5":
        return 0.5
    candidates = []
    brier = float(np.square(probability - labels).mean())
    for cutoff in np.round(np.arange(0.01, 1.0, 0.01), 2):
        scores = classification(labels, probability, float(cutoff))
        precision = float(scores["outbreak_precision"])
        specificity = float(scores["outbreak_specificity"])
        sensitivity = float(scores["outbreak_sensitivity"])
        f1 = float(scores["outbreak_f1"])
        if policy == "f1_min_precision" and precision < 0.50:
            continue
        if policy == "multiobjective":
            if precision < 0.50 or specificity < 0.90:
                continue
            objective = float(
                np.prod(np.clip([precision, specificity, sensitivity, f1, 1 - brier], 1e-9, 1))
                ** 0.2
            )
            key = (objective, f1, sensitivity, specificity, float(cutoff))
        else:
            key = (f1, sensitivity, specificity, float(cutoff))
        candidates.append((key, float(cutoff)))
    return max(candidates)[1] if candidates else 0.5


def crps(observed: np.ndarray, samples: np.ndarray) -> np.ndarray:
    ordered = np.sort(samples, axis=1)
    count = ordered.shape[1]
    weights = 2 * np.arange(1, count + 1) - count - 1
    return (
        np.abs(samples - observed[:, None]).mean(axis=1)
        - (ordered * weights[None]).sum(axis=1) / (count * count)
    )


def coverage(observed: np.ndarray, samples: np.ndarray, level: float) -> float:
    tail = (1 - level) / 2
    lower, upper = np.quantile(samples, [tail, 1 - tail], axis=1)
    return float(((observed >= lower) & (observed <= upper)).mean())


def rescore_draw_model(
    label: str,
    test_frame: pd.DataFrame,
    test_values: dict[int, np.ndarray],
    validation_frame: pd.DataFrame,
    validation_values: dict[int, np.ndarray],
    moh: pd.DataFrame,
    cutoff_policy: str,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = attach_moh(test_frame, moh)
    validation = attach_moh(validation_frame, moh)
    cutoffs: dict[int, float] = {}
    rows = []
    later_rows = []
    output_predictions = []
    for horizon in sorted(test_values):
        valid_part = validation[validation["horizon"].eq(horizon)].reset_index(drop=True)
        valid_matrix = validation_values[horizon]
        valid_threshold = valid_part["outbreak_threshold"].to_numpy(float)
        valid_probability = (valid_matrix > valid_threshold[:, None]).mean(axis=1)
        valid_labels = valid_part["target_cases"].to_numpy(float) > valid_threshold
        cutoff = select_cutoff(valid_labels, valid_probability, cutoff_policy)
        cutoffs[int(horizon)] = cutoff

        part = test[test["horizon"].eq(horizon)].reset_index(drop=True)
        matrix = test_values[horizon]
        threshold = part["outbreak_threshold"].to_numpy(float)
        probability = (matrix > threshold[:, None]).mean(axis=1)
        labels = part["target_cases"].to_numpy(float) > threshold
        population = part["population"].to_numpy(float)
        observed_incidence = part["target_cases"].to_numpy(float) / population * 100_000
        sample_incidence = matrix / population[:, None] * 100_000
        error = sample_incidence.mean(axis=1) - observed_incidence
        scored = part.copy()
        scored["model"] = label
        scored["outbreak_probability"] = probability
        scored["outbreak_decision_threshold"] = cutoff
        output_predictions.append(scored)
        rows.append(
            {
                "model": label,
                "split": "test",
                "horizon": horizon,
                "n": len(part),
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.square(error).mean())),
                "bias": float(error.mean()),
                "crps": float(crps(observed_incidence, sample_incidence).mean()),
                "coverage_50": coverage(observed_incidence, sample_incidence, 0.50),
                "coverage_80": coverage(observed_incidence, sample_incidence, 0.80),
                "coverage_95": coverage(observed_incidence, sample_incidence, 0.95),
                "outbreak_brier": float(np.square(probability - labels).mean()),
                **classification(labels, probability, cutoff),
            }
        )
        later_mask = part["target_date"].between("2023-01-01", "2025-12-01").to_numpy()
        later_part = part.loc[later_mask].reset_index(drop=True)
        later_matrix = matrix[later_mask]
        later_threshold = later_part["outbreak_threshold"].to_numpy(float)
        later_probability = (later_matrix > later_threshold[:, None]).mean(axis=1)
        later_labels = later_part["target_cases"].to_numpy(float) > later_threshold
        later_population = later_part["population"].to_numpy(float)
        later_observed = later_part["target_cases"].to_numpy(float) / later_population * 100_000
        later_incidence = later_matrix / later_population[:, None] * 100_000
        later_error = later_incidence.mean(axis=1) - later_observed
        later_rows.append(
            {
                "model": label,
                "split": "test_2023_2025",
                "horizon": horizon,
                "n": len(later_part),
                "mae": float(np.abs(later_error).mean()),
                "rmse": float(np.sqrt(np.square(later_error).mean())),
                "bias": float(later_error.mean()),
                "crps": float(crps(later_observed, later_incidence).mean()),
                "coverage_50": coverage(later_observed, later_incidence, 0.50),
                "coverage_80": coverage(later_observed, later_incidence, 0.80),
                "coverage_95": coverage(later_observed, later_incidence, 0.95),
                "outbreak_brier": float(np.square(later_probability - later_labels).mean()),
                **classification(later_labels, later_probability, cutoff),
            }
        )
    model_output = output / "models" / label.replace(" ", "_").replace("/", "-")
    model_output.mkdir(parents=True, exist_ok=True)
    pd.concat(output_predictions, ignore_index=True).to_csv(
        model_output / "predictions.csv", index=False
    )
    (model_output / "validation_cutoffs.json").write_text(
        json.dumps(
            {
                "policy": cutoff_policy,
                "selection_split": "2019-01 through 2020-12",
                "cutoffs": cutoffs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(model_output / "metrics.csv", index=False)
    return metrics, pd.DataFrame(later_rows)


QUANTILE_COLUMNS = (
    (0.005, "lower_99CI"),
    (0.025, "lower_95CI"),
    (0.075, "lower_85CI"),
    (0.10, "lower_80CI"),
    (0.125, "lower_75CI"),
    (0.25, "lower_50CI"),
    (0.50, "median"),
    (0.75, "upper_50CI"),
    (0.875, "upper_75CI"),
    (0.90, "upper_80CI"),
    (0.925, "upper_85CI"),
    (0.975, "upper_95CI"),
    (0.995, "upper_99CI"),
)


def reconstructed_quantile_draws(frame: pd.DataFrame, draws: int = 500) -> np.ndarray:
    probabilities = np.asarray([0.0, *[item[0] for item in QUANTILE_COLUMNS], 1.0])
    supplied = frame[[item[1] for item in QUANTILE_COLUMNS]].to_numpy(float)
    lower = np.zeros((len(frame), 1), dtype=float)
    upper = np.maximum(
        supplied[:, -1:] + (supplied[:, -1:] - supplied[:, -2:-1]),
        supplied[:, -1:],
    )
    quantiles = np.concatenate([lower, supplied, upper], axis=1)
    quantiles = np.maximum.accumulate(np.maximum(quantiles, 0), axis=1)
    grid = (np.arange(draws) + 0.5) / draws
    output = np.empty((len(frame), draws), dtype=np.float32)
    for index in range(len(frame)):
        output[index] = np.interp(grid, probabilities, quantiles[index])
    return output


def rescore_quantile_model(
    path: Path, moh: pd.DataFrame, output: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, dtype={"fcode": str})
    frame = frame.rename(
        columns={
            "date": "target_date",
            "vintage_date": "issue_date",
            "obs_dengue_cases": "target_cases",
            "pop_total": "population",
        }
    )
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    frame = frame[frame["target_date"].between("2021-01-01", "2025-12-01")].copy()
    frame = attach_moh(frame, moh)
    label = "Original ensemble"
    rows = []
    later_rows = []
    predictions = []
    for horizon in sorted(frame["horizon"].unique()):
        part = frame[frame["horizon"].eq(horizon)].reset_index(drop=True)
        samples = reconstructed_quantile_draws(part)
        threshold = part["outbreak_threshold"].to_numpy(float)
        probability = (samples > threshold[:, None]).mean(axis=1)
        labels = part["target_cases"].to_numpy(float) > threshold
        population = part["population"].to_numpy(float)
        observed_incidence = part["target_cases"].to_numpy(float) / population * 100_000
        predicted_incidence = part["mean"].to_numpy(float) / population * 100_000
        sample_incidence = samples / population[:, None] * 100_000
        error = predicted_incidence - observed_incidence
        scored = part.copy()
        scored["model"] = label
        scored["outbreak_probability"] = probability
        scored["outbreak_decision_threshold"] = 0.5
        predictions.append(scored)
        rows.append(
            {
                "model": label,
                "split": "test",
                "horizon": int(horizon),
                "n": len(part),
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.square(error).mean())),
                "bias": float(error.mean()),
                "crps": float(crps(observed_incidence, sample_incidence).mean()),
                "coverage_50": float(
                    (
                        (part["target_cases"] >= part["lower_50CI"])
                        & (part["target_cases"] <= part["upper_50CI"])
                    ).mean()
                ),
                "coverage_80": float(
                    (
                        (part["target_cases"] >= part["lower_80CI"])
                        & (part["target_cases"] <= part["upper_80CI"])
                    ).mean()
                ),
                "coverage_95": float(
                    (
                        (part["target_cases"] >= part["lower_95CI"])
                        & (part["target_cases"] <= part["upper_95CI"])
                    ).mean()
                ),
                "outbreak_brier": float(np.square(probability - labels).mean()),
                **classification(labels, probability, 0.5),
            }
        )
        later_mask = part["target_date"].between("2023-01-01", "2025-12-01").to_numpy()
        later_part = part.loc[later_mask].reset_index(drop=True)
        later_samples = samples[later_mask]
        later_threshold = later_part["outbreak_threshold"].to_numpy(float)
        later_probability = (later_samples > later_threshold[:, None]).mean(axis=1)
        later_labels = later_part["target_cases"].to_numpy(float) > later_threshold
        later_population = later_part["population"].to_numpy(float)
        later_observed = later_part["target_cases"].to_numpy(float) / later_population * 100_000
        later_predicted = later_part["mean"].to_numpy(float) / later_population * 100_000
        later_incidence = later_samples / later_population[:, None] * 100_000
        later_error = later_predicted - later_observed
        later_rows.append(
            {
                "model": label,
                "split": "test_2023_2025",
                "horizon": int(horizon),
                "n": len(later_part),
                "mae": float(np.abs(later_error).mean()),
                "rmse": float(np.sqrt(np.square(later_error).mean())),
                "bias": float(later_error.mean()),
                "crps": float(crps(later_observed, later_incidence).mean()),
                "coverage_50": float(((later_part["target_cases"] >= later_part["lower_50CI"]) & (later_part["target_cases"] <= later_part["upper_50CI"])).mean()),
                "coverage_80": float(((later_part["target_cases"] >= later_part["lower_80CI"]) & (later_part["target_cases"] <= later_part["upper_80CI"])).mean()),
                "coverage_95": float(((later_part["target_cases"] >= later_part["lower_95CI"]) & (later_part["target_cases"] <= later_part["upper_95CI"])).mean()),
                "outbreak_brier": float(np.square(later_probability - later_labels).mean()),
                **classification(later_labels, later_probability, 0.5),
            }
        )
    model_output = output / "models" / "original_ensemble_549_quantile_reconstruction"
    model_output.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions, ignore_index=True).to_csv(
        model_output / "predictions.csv", index=False
    )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(model_output / "metrics.csv", index=False)
    return metrics, pd.DataFrame(later_rows)


def summarize(by_horizon: pd.DataFrame) -> pd.DataFrame:
    summary = by_horizon.groupby("model", as_index=False)[list(METRIC_COLUMNS)].mean()
    for column in METRIC_COLUMNS:
        summary[f"rank_{column}"] = summary[column].rank(
            ascending=column in LOSS_COLUMNS, method="min"
        )
    ranks = [f"rank_{column}" for column in METRIC_COLUMNS]
    summary["mean_metric_rank"] = summary[ranks].mean(axis=1)
    summary["worst_metric_rank"] = summary[ranks].max(axis=1)
    return summary.sort_values(["mean_metric_rank", "worst_metric_rank"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=ROOT / "code/data/raw/operational_2026/commune_monthly_thresholds.csv",
    )
    parser.add_argument(
        "--external-quantiles",
        type=Path,
        default=ROOT / "code/data/raw/operational_2026/original_commune_forecast_quantiles.csv",
    )
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results/moh_rescored")
    parser.add_argument("--current", type=Path, default=ROOT / "results/current")
    args = parser.parse_args()
    moh = load_moh(args.thresholds)
    args.output.mkdir(parents=True, exist_ok=True)

    specifications = [
        ("TCN-GNN Base", "npz", "TCN-GNN-BASE", "fixed_0.5"),
        ("DeepST-Alert (proposed deep)", "npz", "DEEPST-ALERT", "fixed_0.5"),
        ("TCN-GNN Deep Ensemble (variant)", "npz", "TCN-GNN-DEEP-ENSEMBLE", "f1_min_precision"),
        ("Representation-INLA (variant)", "binary", "REPRESENTATION-INLA", "fixed_0.5"),
        ("DeepST-INLA (proposed hybrid)", "npz", "DEEPST-INLA", "multiobjective"),
        ("INLA-TS", "binary", "INLA-ONLY", "fixed_0.5"),
    ]
    results = []
    later_results = []
    for label, kind, relative, policy in specifications:
        if kind == "npz":
            directory = args.artifact_root / relative
            test_frame, test_values = load_npz_split(directory, "test")
            if relative == "TCN-GNN-DEEP-ENSEMBLE":
                validation_dir = args.artifact_root / "TCN-GNN-BASE"
                validation_frame, validation_values = load_npz_split(
                    validation_dir, "validation"
                )
            elif relative == "DEEPST-INLA":
                validation_frame, validation_values = load_lognormal_validation(
                    directory
                )
            else:
                validation_dir = directory
                validation_frame, validation_values = load_npz_split(
                    validation_dir, "validation"
                )
        elif relative == "REPRESENTATION-INLA":
            test_frame, test_values = load_binary_split(directory := args.artifact_root / relative / "test/combined")
            validation_frame, validation_values = load_binary_split(args.artifact_root / relative / "validation/combined")
        else:
            test_frame, test_values = load_binary_split(args.artifact_root / "INLA-ONLY-test/INLA-TS")
            validation_frame, validation_values = load_binary_split(args.artifact_root / "INLA-ONLY-validation/INLA-TS")
        full_metrics, later_metrics = rescore_draw_model(
            label,
            test_frame,
            test_values,
            validation_frame,
            validation_values,
            moh,
            policy,
            args.output,
        )
        results.append(full_metrics)
        later_results.append(later_metrics)
    full_metrics, later_metrics = rescore_quantile_model(
        args.external_quantiles, moh, args.output
    )
    results.append(full_metrics)
    later_results.append(later_metrics)
    by_horizon = pd.concat(results, ignore_index=True).sort_values(["model", "horizon"])
    mean = summarize(by_horizon)
    later_by_horizon = pd.concat(later_results, ignore_index=True).sort_values(
        ["model", "horizon"]
    )
    later_mean = summarize(later_by_horizon)

    args.output.mkdir(parents=True, exist_ok=True)
    by_horizon.to_csv(args.output / "final_metrics_by_horizon.csv", index=False)
    mean.to_csv(args.output / "final_metrics_mean_H1_H3.csv", index=False)
    later_by_horizon.to_csv(
        args.output / "final_metrics_by_horizon_2023_2025.csv", index=False
    )
    later_mean.to_csv(
        args.output / "final_metrics_mean_H1_H3_2023_2025.csv", index=False
    )
    provenance = {
        "threshold_definition": "MOH same-calendar-month mean + 2 sample SD after excluding a candidate outbreak year",
        "threshold_file": str(args.thresholds),
        "source_artifacts": str(args.artifact_root),
        "count_models_refit": False,
        "probabilities": "recomputed from retained predictive draws against current MOH thresholds",
        "decision_cutoffs": "2019-2020 validation only; fixed at 0.5 where specified",
        "external_549_model": {
            "source": str(args.external_quantiles),
            "limitation": "No draws were supplied; outbreak probabilities and CRPS use deterministic interpolation of the supplied quantiles.",
            "target_coverage": "H1 2021-01 to 2025-10; H2 2021-01 to 2025-11; H3 2021-01 to 2025-12",
        },
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    archive = args.current / "pre_moh_correction_20261007"
    archive.mkdir(parents=True, exist_ok=True)
    for name in (
        "final_metrics_by_horizon.csv",
        "final_metrics_mean_H1_H3.csv",
        "final_metrics_by_horizon_2023_2025.csv",
        "final_metrics_mean_H1_H3_2023_2025.csv",
    ):
        destination = args.current / name
        if destination.exists() and not (archive / name).exists():
            shutil.copy2(destination, archive / name)
        shutil.copy2(args.output / name, destination)
    shutil.copy2(
        args.output / "provenance.json", args.current / "moh_rescoring_provenance.json"
    )
    print(mean.to_string(index=False))


if __name__ == "__main__":
    main()
