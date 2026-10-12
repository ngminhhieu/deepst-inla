from __future__ import annotations

import numpy as np
import pandas as pd


OUTBREAK_DECISION_THRESHOLD = 0.50


def score_predictions(
    predictions: pd.DataFrame,
    samples: np.ndarray,
    *,
    decision_threshold: float | dict[int, float] = OUTBREAK_DECISION_THRESHOLD,
    split: str = "test",
) -> pd.DataFrame:
    rows = []
    for horizon in sorted(predictions["horizon"].unique()):
        frame = predictions[predictions["horizon"] == horizon].reset_index(drop=True)
        horizon_index = sorted(predictions["horizon"].unique()).index(horizon)
        horizon_samples = samples[:, :, horizon_index, :].reshape(-1, samples.shape[-1])
        observed = frame["target_cases"].to_numpy(float)
        population = frame["population"].to_numpy(float)
        sample_incidence = horizon_samples / population[:, None] * 100_000.0
        observed_incidence = observed / population * 100_000.0
        predicted_incidence = sample_incidence.mean(axis=1)
        error = predicted_incidence - observed_incidence
        outbreak_observed = (
            observed > frame["outbreak_threshold"].to_numpy(float)
        ).astype(float)
        outbreak_probability = frame["outbreak_probability"].to_numpy(float)
        horizon_decision_threshold = (
            float(decision_threshold[int(horizon)])
            if isinstance(decision_threshold, dict)
            else float(decision_threshold)
        )
        classification = outbreak_classification_metrics(
            observed,
            frame["outbreak_threshold"].to_numpy(float),
            outbreak_probability,
            decision_threshold=horizon_decision_threshold,
        )
        rows.append(
            {
                "model": frame["model"].iloc[0],
                "split": split,
                "horizon": int(horizon),
                "n": len(frame),
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.square(error).mean())),
                "bias": float(error.mean()),
                "crps": float(_crps(observed_incidence, sample_incidence).mean()),
                "coverage_50": _coverage(
                    observed_incidence, sample_incidence, 0.50
                ),
                "coverage_80": _coverage(
                    observed_incidence, sample_incidence, 0.80
                ),
                "coverage_95": _coverage(
                    observed_incidence, sample_incidence, 0.95
                ),
                "outbreak_brier": float(
                    np.square(outbreak_probability - outbreak_observed).mean()
                ),
                **classification,
            }
        )
    return pd.DataFrame(rows)


def outbreak_classification_metrics(
    observed_cases: np.ndarray,
    outbreak_threshold: np.ndarray,
    outbreak_probability: np.ndarray,
    decision_threshold: float = OUTBREAK_DECISION_THRESHOLD,
) -> dict[str, float | int]:
    """Score binary alerts derived from probabilistic outbreak forecasts.

    Observed and predicted outbreaks use the same case-count threshold. The
    prediction becomes an alert when its probability of exceeding that count
    threshold is at least ``decision_threshold``.
    """
    observed = np.asarray(observed_cases, dtype=float) > np.asarray(
        outbreak_threshold, dtype=float
    )
    predicted = np.asarray(outbreak_probability, dtype=float) >= decision_threshold
    tp = int(np.sum(observed & predicted))
    fp = int(np.sum(~observed & predicted))
    fn = int(np.sum(observed & ~predicted))
    tn = int(np.sum(~observed & ~predicted))
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    sensitivity = recall
    specificity = _safe_ratio(tn, tn + fp)
    return {
        "outbreak_decision_threshold": float(decision_threshold),
        "outbreak_observed_positive": int(observed.sum()),
        "outbreak_predicted_positive": int(predicted.sum()),
        "outbreak_tp": tp,
        "outbreak_fp": fp,
        "outbreak_fn": fn,
        "outbreak_tn": tn,
        "outbreak_accuracy": _safe_ratio(tp + tn, len(observed)),
        "outbreak_precision": precision,
        "outbreak_specificity": specificity,
        "outbreak_recall": recall,
        "outbreak_sensitivity": sensitivity,
        # Kept as a compatibility alias; specificity and true-negative rate are identical.
        "outbreak_true_negative_rate": specificity,
        "outbreak_balanced_accuracy": (sensitivity + specificity) / 2.0,
        "outbreak_f1": _safe_ratio(2.0 * precision * recall, precision + recall),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _coverage(
    observed: np.ndarray, samples: np.ndarray, level: float
) -> float:
    tail = (1.0 - level) / 2.0
    lower = np.quantile(samples, tail, axis=1)
    upper = np.quantile(samples, 1.0 - tail, axis=1)
    return float(((observed >= lower) & (observed <= upper)).mean())


def _crps(observed: np.ndarray, samples: np.ndarray) -> np.ndarray:
    ordered = np.sort(samples, axis=1)
    count = ordered.shape[1]
    weights = 2 * np.arange(1, count + 1) - count - 1
    pairwise_term = (ordered * weights[None]).sum(axis=1) / (count * count)
    return np.abs(samples - observed[:, None]).mean(axis=1) - pairwise_term
