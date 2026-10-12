#!/usr/bin/env python3
"""Calibrate, score, resample, and plot the province cross-scale experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score


MODEL_ORDER = ["INLA-TS", "TCN-GNN", "DeepST-Alert", "DeepST-INLA"]
COMMUNE_NAMES = {
    "INLA-TS": "INLA-TS",
    "TCN-GNN Base": "TCN-GNN",
    "DeepST-Alert (proposed deep)": "DeepST-Alert",
    "DeepST-INLA (proposed hybrid)": "DeepST-INLA",
}
KEYS = ["fcode", "issue_date", "target_date", "horizon"]

mpl.rcParams.update({
    "font.size": 12.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 9.5,
})


def fit_platt(probability: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Regularised, monotone Platt fit robust to a small separated validation set."""
    clipped = np.clip(np.asarray(probability, float), 1e-4, 1 - 1e-4)
    x = np.log(clipped / (1 - clipped))
    y = np.asarray(labels, float)

    def objective(parameters: np.ndarray) -> float:
        linear = np.clip(parameters[0] + parameters[1] * x, -30, 30)
        nll = np.mean(np.logaddexp(0, linear) - y * linear)
        return float(nll + 0.01 * (parameters[1] - 1.0) ** 2)

    result = minimize(
        objective, np.array([0.0, 1.0]), method="L-BFGS-B",
        bounds=((-10.0, 10.0), (0.05, 10.0)),
    )
    if not result.success:
        raise RuntimeError(f"Platt calibration failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


def apply_platt(probability: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, float), 1e-4, 1 - 1e-4)
    logit = np.log(clipped / (1 - clipped))
    linear = np.clip(intercept + slope * logit, -30, 30)
    return 1 / (1 + np.exp(-linear))


def classification(labels: np.ndarray, probability: np.ndarray, cutoff: float) -> dict[str, float]:
    labels = np.asarray(labels, bool)
    alerts = np.asarray(probability, float) >= cutoff
    tp, fp = np.sum(labels & alerts), np.sum(~labels & alerts)
    fn, tn = np.sum(labels & ~alerts), np.sum(~labels & ~alerts)
    ratio = lambda a, b: float(a / b) if b else 0.0
    sensitivity = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    precision = ratio(tp, tp + fp)
    return {
        "outbreak_observed_positive": int(labels.sum()),
        "outbreak_predicted_positive": int(alerts.sum()),
        "outbreak_tp": int(tp), "outbreak_fp": int(fp),
        "outbreak_fn": int(fn), "outbreak_tn": int(tn),
        "outbreak_precision": precision,
        "outbreak_specificity": specificity,
        "outbreak_sensitivity": sensitivity,
        "outbreak_recall": sensitivity,
        "outbreak_true_negative_rate": specificity,
        "outbreak_accuracy": ratio(tp + tn, len(labels)),
        "outbreak_balanced_accuracy": (sensitivity + specificity) / 2,
        "outbreak_f1": ratio(2 * precision * sensitivity, precision + sensitivity),
        "false_alert_rate": 1 - specificity,
    }


def select_f1_cutoff(labels: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.linspace(0.01, 0.99, 99)
    scored = []
    for cutoff in candidates:
        metrics = classification(labels, probability, float(cutoff))
        scored.append((metrics["outbreak_f1"], metrics["outbreak_specificity"], cutoff))
    return float(max(scored)[2])


def threshold_for_constraint(
    labels: np.ndarray, probability: np.ndarray, *, specificity: float | None = None,
    sensitivity: float | None = None,
) -> tuple[float, dict[str, float]]:
    candidates = np.unique(np.r_[0.0, probability, 1.0])
    scored = [(float(cut), classification(labels, probability, float(cut))) for cut in candidates]
    if specificity is not None:
        eligible = [row for row in scored if row[1]["outbreak_specificity"] >= specificity]
        winner = max(eligible, key=lambda row: (row[1]["outbreak_sensitivity"], row[0]))
    else:
        eligible = [row for row in scored if row[1]["outbreak_sensitivity"] >= float(sensitivity)]
        winner = max(eligible, key=lambda row: (row[1]["outbreak_specificity"], -row[0]))
    return winner


def load_model_frames(models_root: Path) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    locations = {
        "TCN-GNN": (
            models_root / "TCN-GNN/validation_predictions.csv",
            models_root / "TCN-GNN/predictions.csv", models_root / "TCN-GNN/metrics.csv"),
        "DeepST-Alert": (
            models_root / "DeepST-Alert/validation_predictions.csv",
            models_root / "DeepST-Alert/predictions.csv", models_root / "DeepST-Alert/metrics.csv"),
        "INLA-TS": (
            models_root / "INLA-validation/INLA-TS/predictions.csv",
            models_root / "INLA-test/INLA-TS/predictions.csv",
            models_root / "INLA-test/INLA-TS/metrics.csv"),
        "DeepST-INLA": (
            models_root / "DeepST-INLA/validation_predictions.csv",
            models_root / "DeepST-INLA/predictions.csv", models_root / "DeepST-INLA/metrics.csv"),
    }
    return {
        model: tuple(pd.read_csv(path, dtype={"fcode": str}) for path in paths)
        for model, paths in locations.items()
    }


def calibration_diagnostic(labels: np.ndarray, probability: np.ndarray) -> tuple[float, float, float]:
    intercept, slope = fit_platt(probability, labels)
    order = np.argsort(probability)
    bins = np.array_split(order, 5)
    ece = sum(
        len(index) / len(labels) * abs(labels[index].mean() - probability[index].mean())
        for index in bins if len(index)
    )
    return intercept, slope, float(ece)


def prepare_predictions(frames):
    outputs, validation_outputs, policies, calibration_rows = [], [], [], []
    for model in MODEL_ORDER:
        validation, test, _ = frames[model]
        for frame in (validation, test):
            frame["issue_date"] = pd.to_datetime(frame.issue_date)
            frame["target_date"] = pd.to_datetime(frame.target_date)
            frame["observed_outbreak"] = frame.target_cases > frame.outbreak_threshold
        for horizon in (1, 2, 3):
            valid = validation[validation.horizon.eq(horizon)].copy()
            target = test[test.horizon.eq(horizon)].copy()
            if model == "DeepST-INLA":
                intercept, slope = 0.0, 1.0
                valid_probability = valid.outbreak_probability.to_numpy(float)
                test_probability = target.outbreak_probability.to_numpy(float)
                cutoff = float(valid.outbreak_decision_threshold.iloc[0])
                calibration_method = "embedded validation Platt policy"
            else:
                intercept, slope = fit_platt(
                    valid.outbreak_probability.to_numpy(float), valid.observed_outbreak.to_numpy(float)
                )
                valid_probability = apply_platt(valid.outbreak_probability, intercept, slope)
                test_probability = apply_platt(target.outbreak_probability, intercept, slope)
                cutoff = select_f1_cutoff(valid.observed_outbreak.to_numpy(), valid_probability)
                calibration_method = "validation Platt; maximum validation F1 cutoff"
            valid["raw_outbreak_probability"] = valid.outbreak_probability
            target["raw_outbreak_probability"] = target.outbreak_probability
            valid["outbreak_probability"] = valid_probability
            target["outbreak_probability"] = test_probability
            valid["outbreak_decision_threshold"] = cutoff
            target["outbreak_decision_threshold"] = cutoff
            valid["model"] = target["model"] = model
            validation_outputs.append(valid)
            outputs.append(target)
            policies.append({
                "model": model, "horizon": horizon, "calibration_intercept": intercept,
                "calibration_slope": slope, "decision_threshold": cutoff,
                "selection_split": "2019-01 to 2020-12", "method": calibration_method,
            })
            diag_intercept, diag_slope, ece = calibration_diagnostic(
                target.observed_outbreak.to_numpy(float), test_probability
            )
            calibration_rows.append({
                "scale": "Province", "model": model, "horizon": horizon, "n": len(target),
                "outbreak_prevalence": target.observed_outbreak.mean(),
                "brier": np.mean((test_probability - target.observed_outbreak) ** 2),
                "test_calibration_intercept": diag_intercept,
                "test_calibration_slope": diag_slope, "ece_5_equal_count_bins": ece,
                "validation_platt_intercept": intercept, "validation_platt_slope": slope,
                "locked_decision_threshold": cutoff,
            })
    return (
        pd.concat(outputs, ignore_index=True),
        pd.concat(validation_outputs, ignore_index=True),
        pd.DataFrame(policies), pd.DataFrame(calibration_rows),
    )


def metrics_tables(predictions: pd.DataFrame, frames):
    rows = []
    for model in MODEL_ORDER:
        source_metrics = frames[model][2].copy()
        for horizon in (1, 2, 3):
            part = predictions[(predictions.model == model) & predictions.horizon.eq(horizon)]
            base = source_metrics[source_metrics.horizon.eq(horizon)].iloc[0].to_dict()
            labels = part.observed_outbreak.to_numpy(bool)
            probability = part.outbreak_probability.to_numpy(float)
            cutoff = float(part.outbreak_decision_threshold.iloc[0])
            base.update(classification(labels, probability, cutoff))
            base.update({
                "model": model, "scale": "Province", "horizon": horizon, "n": len(part),
                "outbreak_brier": float(np.mean((probability - labels) ** 2)),
                "outbreak_prevalence": float(labels.mean()),
                "outbreak_decision_threshold": cutoff,
                "mean_observed_incidence": float(
                    np.mean(part.target_cases / part.population * 100_000)
                ),
            })
            base["normalized_mae"] = base["mae"] / base["mean_observed_incidence"]
            base["normalized_crps"] = base["crps"] / base["mean_observed_incidence"]
            rows.append(base)
    by_horizon = pd.DataFrame(rows)
    numeric = [
        "mae", "rmse", "crps", "outbreak_brier", "outbreak_precision",
        "outbreak_specificity", "outbreak_sensitivity", "outbreak_f1",
        "false_alert_rate", "outbreak_prevalence", "mean_observed_incidence",
        "normalized_mae", "normalized_crps",
    ]
    mean = by_horizon.groupby(["scale", "model"], as_index=False)[numeric].mean()
    return by_horizon, mean


def matched_operating_points(predictions: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        for horizon in (1, 2, 3):
            valid = validation[(validation.model == model) & validation.horizon.eq(horizon)]
            test = predictions[(predictions.model == model) & predictions.horizon.eq(horizon)]
            vy, vp = valid.observed_outbreak.to_numpy(bool), valid.outbreak_probability.to_numpy(float)
            ty, tp = test.observed_outbreak.to_numpy(bool), test.outbreak_probability.to_numpy(float)
            row = {
                "scale": "Province", "model": model, "horizon": horizon,
                "test_auprc": average_precision_score(ty, tp),
            }
            for target in (0.90, 0.95):
                cutoff, achieved = threshold_for_constraint(vy, vp, specificity=target)
                scored = classification(ty, tp, cutoff)
                suffix = str(int(target * 100))
                row[f"validation_cutoff_at_specificity_{suffix}"] = cutoff
                row[f"validation_specificity_at_cutoff_{suffix}"] = achieved["outbreak_specificity"]
                row[f"test_sensitivity_at_specificity_{suffix}"] = scored["outbreak_sensitivity"]
                row[f"test_specificity_at_specificity_{suffix}"] = scored["outbreak_specificity"]
            cutoff, achieved = threshold_for_constraint(vy, vp, sensitivity=0.80)
            scored = classification(ty, tp, cutoff)
            row["validation_cutoff_at_sensitivity_80"] = cutoff
            row["validation_sensitivity_at_cutoff_80"] = achieved["outbreak_sensitivity"]
            row["test_specificity_at_sensitivity_80"] = scored["outbreak_specificity"]
            row["test_sensitivity_at_sensitivity_80"] = scored["outbreak_sensitivity"]
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_pairwise(predictions: pd.DataFrame, replicates: int = 2000) -> pd.DataFrame:
    wide = {model: predictions[predictions.model.eq(model)].sort_values(KEYS).reset_index(drop=True)
            for model in MODEL_ORDER}
    reference = wide["DeepST-INLA"]
    for model, frame in wide.items():
        if not frame[KEYS].equals(reference[KEYS]):
            raise ValueError(f"Prediction keys do not align for {model}")
    months = reference.target_date.dt.to_period("M").astype(str).to_numpy()
    unique_months = np.unique(months)
    rng = np.random.default_rng(9411)
    metrics = ["brier", "sensitivity", "specificity", "precision", "f1", "false_alert_rate"]

    def values(frame, indices):
        y = frame.observed_outbreak.to_numpy(bool)[indices]
        p = frame.outbreak_probability.to_numpy(float)[indices]
        cutoff = frame.outbreak_decision_threshold.to_numpy(float)[indices]
        alert = p >= cutoff
        tp, fp = np.sum(y & alert), np.sum(~y & alert)
        fn, tn = np.sum(y & ~alert), np.sum(~y & ~alert)
        ratio = lambda a, b: float(a / b) if b else 0.0
        sensitivity, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
        precision = ratio(tp, tp + fp)
        return np.array([
            np.mean((p - y) ** 2), sensitivity, specificity, precision,
            ratio(2 * precision * sensitivity, precision + sensitivity), 1 - specificity,
        ])

    rows = []
    all_indices = np.arange(len(reference))
    for comparator in ("TCN-GNN", "INLA-TS", "DeepST-Alert"):
        observed = values(reference, all_indices) - values(wide[comparator], all_indices)
        draws = np.empty((replicates, len(metrics)))
        for replicate in range(replicates):
            sampled = rng.choice(unique_months, size=len(unique_months), replace=True)
            indices = np.concatenate([np.flatnonzero(months == month) for month in sampled])
            draws[replicate] = values(reference, indices) - values(wide[comparator], indices)
        for index, metric in enumerate(metrics):
            low, high = np.quantile(draws[:, index], [0.025, 0.975])
            rows.append({
                "scale": "Province", "model_a": "DeepST-INLA", "model_b": comparator,
                "difference": "model_a_minus_model_b", "metric": metric,
                "estimate": observed[index], "ci_lower": low, "ci_upper": high,
                "bootstrap_replicates": replicates,
                "block": "target calendar month; all provinces and horizons retained",
                "interval_excludes_zero": bool(low > 0 or high < 0),
            })
    return pd.DataFrame(rows)


def plot_reliability(predictions: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), sharex=True, sharey=True)
    colors = dict(zip(MODEL_ORDER, ["#5B4B8A", "#0072B2", "#D55E00", "#009E73"]))
    for horizon, ax in zip((1, 2, 3), axes):
        ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--")
        for model in MODEL_ORDER:
            part = predictions[(predictions.model == model) & predictions.horizon.eq(horizon)]
            order = np.argsort(part.outbreak_probability.to_numpy())
            bins = np.array_split(order, 5)
            x = [part.outbreak_probability.to_numpy()[idx].mean() for idx in bins]
            y = [part.observed_outbreak.to_numpy()[idx].mean() for idx in bins]
            ax.plot(x, y, marker="o", ms=3, lw=1, color=colors[model], label=model)
        ax.set_title(f"H{horizon}")
        ax.set_xlabel("Mean predicted probability")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Observed outbreak frequency")
    axes[-1].legend(frameon=False, fontsize=9.5, loc="lower right")
    fig.suptitle("Province-level probability reliability (five equal-count bins)", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def cross_scale_source(commune_metrics: Path, province_mean: pd.DataFrame) -> pd.DataFrame:
    commune = pd.read_csv(commune_metrics)
    commune = commune[commune.model.isin(COMMUNE_NAMES)].copy()
    commune["model"] = commune.model.map(COMMUNE_NAMES)
    commune["scale"] = "Commune"
    commune["false_alert_rate"] = 1 - commune.outbreak_specificity
    columns = ["scale", "model", "outbreak_brier", "outbreak_sensitivity",
               "outbreak_specificity", "outbreak_precision", "outbreak_f1", "false_alert_rate"]
    return pd.concat([commune[columns], province_mean[columns]], ignore_index=True)


def plot_cross_scale(source: pd.DataFrame, output: Path) -> None:
    colors = dict(zip(MODEL_ORDER, ["#5B4B8A", "#0072B2", "#D55E00", "#009E73"]))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), sharex=True, sharey=True)
    for ax, scale in zip(axes, ("Commune", "Province")):
        part = source[source.scale.eq(scale)]
        for _, row in part.iterrows():
            ax.scatter(row.false_alert_rate, row.outbreak_sensitivity, s=48,
                       color=colors[row.model], label=row.model, zorder=3)
            ax.annotate(row.model, (row.false_alert_rate, row.outbreak_sensitivity),
                        xytext=(4, 3), textcoords="offset points", fontsize=9.5)
        ax.set_title(f"{chr(97 + len(ax.figure.axes) - 2)}" if False else scale)
        ax.set_xlabel("False-alert rate (1 − specificity)")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Sensitivity")
    axes[0].set_xlim(-0.015, max(0.25, source.false_alert_rate.max() + 0.03))
    axes[0].set_ylim(0, 1.02)
    fig.suptitle("Cross-scale outbreak-alert operating behaviour", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--commune-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_model_frames(args.models_root)
    predictions, validation, policies, calibration = prepare_predictions(frames)
    by_horizon, mean = metrics_tables(predictions, frames)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False, date_format="%Y-%m-%d")
    validation.to_csv(args.output_dir / "validation_predictions.csv", index=False, date_format="%Y-%m-%d")
    policies.to_csv(args.output_dir / "validation_selected_policies.csv", index=False)
    by_horizon.to_csv(args.output_dir / "metrics_by_horizon.csv", index=False)
    mean.to_csv(args.output_dir / "metrics_mean_H1_H3.csv", index=False)
    calibration.to_csv(args.output_dir / "calibration_diagnostics.csv", index=False)
    matched_operating_points(predictions, validation).to_csv(
        args.output_dir / "matched_operating_points.csv", index=False)
    bootstrap_pairwise(predictions).to_csv(args.output_dir / "bootstrap_pairwise.csv", index=False)
    plot_reliability(predictions, args.output_dir / "province_reliability_diagrams.pdf")
    source = cross_scale_source(args.commune_metrics, mean)
    source.to_csv(args.output_dir / "FigS_cross_scale_operating_points_source.csv", index=False)
    plot_cross_scale(source, args.output_dir / "FigS_cross_scale_operating_points.pdf")
    verification = {
        "calibration_selection_rows": int(len(validation)),
        "test_rows": int(len(predictions)),
        "selection_uses_test_outcomes": False,
        "neural_validation_forecast_source": "TCN-GNN-selection and DeepST-Alert-selection",
        "neural_validation_models_refit_on_validation": False,
        "neural_test_models_refit_on_train_plus_validation": True,
        "selection_artifacts_have_refit_history": bool(
            any((args.models_root / f"{name}-selection/selection_history.csv").exists()
                for name in ("TCN-GNN", "DeepST-Alert"))
        ),
        "final_artifacts_have_refit_history": bool(
            all((args.models_root / f"{name}/selection_history.csv").exists()
                for name in ("TCN-GNN", "DeepST-Alert"))
        ),
        "split_windows": {"train": "2010-01/2018-12", "validation": "2019-01/2020-12", "test": "2021-01/2025-12"},
        "test_target_date_min": str(predictions.target_date.min().date()),
        "test_target_date_max": str(predictions.target_date.max().date()),
        "outbreak_rule": (
            "MOH same-calendar-month mean + 2 sample SD after candidate "
            "outbreak-year exclusion and older-year replacement"
        ),
    }
    (args.output_dir / "leakage_verification.json").write_text(json.dumps(verification, indent=2) + "\n")


if __name__ == "__main__":
    main()
