#!/usr/bin/env python3
"""Tune the MOH alert layer and stack it with deep forecast members.

All weights, calibration coefficients, and decision cutoffs are selected on
the 2019--2020 validation split.  The locked 2021--2025 test outcomes are used
only after the policies have been fixed.  The count distribution remains the
DeepST-INLA distribution, preserving its count-forecast performance while the
alert layer combines MOH exceedance evidence from several deep models.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rescore_current_results_with_moh import (
    ARTIFACT_ROOT,
    METRIC_COLUMNS,
    attach_moh,
    classification,
    coverage,
    crps,
    load_lognormal_validation,
    load_moh,
    load_npz_split,
    summarize,
)


ROOT = Path(__file__).resolve().parents[2]
KEYS = ["fcode", "issue_date", "target_date", "horizon"]


def align(
    template: pd.DataFrame,
    source: pd.DataFrame,
    values: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    output = {}
    for horizon in sorted(values):
        target = template[template["horizon"].eq(horizon)].reset_index(drop=True)
        member = source[source["horizon"].eq(horizon)].reset_index(drop=True)
        for column in ("issue_date", "target_date"):
            target[column] = pd.to_datetime(target[column])
            member[column] = pd.to_datetime(member[column])
        target_index = pd.MultiIndex.from_frame(target[KEYS])
        member_index = pd.MultiIndex.from_frame(member[KEYS])
        positions = member_index.get_indexer(target_index)
        if np.any(positions < 0):
            raise ValueError(f"Member rows do not align at H{horizon}")
        output[int(horizon)] = values[int(horizon)][positions]
    return output


def compositions(total: int, members: int):
    if members == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for remainder in compositions(total - first, members - 1):
            yield (first, *remainder)


def fit_platt(probability: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-5, 1 - 1e-5)
    logit = np.log(clipped / (1 - clipped))
    design = np.column_stack((np.ones(len(logit)), logit))
    coefficients = np.asarray([0.0, 1.0])
    penalty = np.diag([0.0, 1e-3])
    for _ in range(50):
        linear = np.clip(design @ coefficients, -30, 30)
        fitted = 1 / (1 + np.exp(-linear))
        variance = np.clip(fitted * (1 - fitted), 1e-6, None)
        gradient = design.T @ (fitted - labels) + penalty @ coefficients
        hessian = design.T @ (variance[:, None] * design) + penalty
        update = np.linalg.solve(hessian, gradient)
        coefficients -= update
        if np.max(np.abs(update)) < 1e-8:
            break
    return float(coefficients[0]), float(coefficients[1])


def apply_platt(probability: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    clipped = np.clip(probability, 1e-5, 1 - 1e-5)
    logit = np.log(clipped / (1 - clipped))
    linear = np.clip(intercept + slope * logit, -30, 30)
    return 1 / (1 + np.exp(-linear))


def select_policy(
    labels: np.ndarray,
    member_probabilities: list[np.ndarray],
    member_names: list[str],
    *,
    weight_units: int,
    minimum_precision: float,
    minimum_first_weight: float = 0.0,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidates = []
    for composition in compositions(weight_units, len(member_probabilities)):
        weights = np.asarray(composition, dtype=float) / weight_units
        if weights[0] < minimum_first_weight:
            continue
        raw = sum(
            weight * probability
            for weight, probability in zip(weights, member_probabilities)
        )
        intercept, slope = fit_platt(raw, labels)
        calibrated = apply_platt(raw, intercept, slope)
        brier = float(np.square(calibrated - labels).mean())
        best = None
        for cutoff in np.round(np.arange(0.05, 0.81, 0.01), 2):
            scores = classification(labels, calibrated, float(cutoff))
            precision = float(scores["outbreak_precision"])
            if precision < minimum_precision:
                continue
            row = {
                "member_names": member_names,
                "weights": weights.tolist(),
                "platt_intercept": intercept,
                "platt_slope": slope,
                "decision_threshold": float(cutoff),
                "brier": brier,
                "precision": precision,
                "specificity": float(scores["outbreak_specificity"]),
                "sensitivity": float(scores["outbreak_sensitivity"]),
                "f1": float(scores["outbreak_f1"]),
            }
            key = (
                row["f1"],
                row["sensitivity"],
                -row["brier"],
                row["specificity"],
            )
            if best is None or key > best[0]:
                best = (key, row)
        if best is not None:
            candidates.append(best[1])
    if not candidates:
        raise ValueError("No validation policy met the minimum precision")
    winner = max(
        candidates,
        key=lambda row: (
            row["f1"], row["sensitivity"], -row["brier"], row["specificity"]
        ),
    )
    return winner, candidates


def apply_policy(
    member_probabilities: list[np.ndarray], policy: dict[str, object]
) -> np.ndarray:
    raw = sum(
        float(weight) * probability
        for weight, probability in zip(policy["weights"], member_probabilities)
    )
    return apply_platt(
        raw, float(policy["platt_intercept"]), float(policy["platt_slope"])
    )


def score(
    label: str,
    frame: pd.DataFrame,
    samples: dict[int, np.ndarray],
    probabilities: dict[int, np.ndarray],
    policies: dict[int, dict[str, object]],
    start: str,
    end: str,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    predictions = []
    for horizon in sorted(samples):
        source = frame[frame["horizon"].eq(horizon)].reset_index(drop=True)
        period = pd.to_datetime(source["target_date"]).between(start, end).to_numpy()
        part = source.loc[period].reset_index(drop=True)
        values = samples[horizon][period]
        probability = probabilities[horizon][period]
        threshold = part["outbreak_threshold"].to_numpy(float)
        observed_cases = part["target_cases"].to_numpy(float)
        population = part["population"].to_numpy(float)
        observed = observed_cases / population * 100_000
        incidence = values / population[:, None] * 100_000
        error = incidence.mean(axis=1) - observed
        labels = observed_cases > threshold
        cutoff = float(policies[horizon]["decision_threshold"])
        output = part.copy()
        output["model"] = label
        output["outbreak_probability"] = probability
        output["outbreak_decision_threshold"] = cutoff
        predictions.append(output)
        rows.append(
            {
                "model": label,
                "split": split,
                "horizon": horizon,
                "n": len(part),
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.square(error).mean())),
                "bias": float(error.mean()),
                "crps": float(crps(observed, incidence).mean()),
                "coverage_50": coverage(observed, incidence, 0.50),
                "coverage_80": coverage(observed, incidence, 0.80),
                "coverage_95": coverage(observed, incidence, 0.95),
                "outbreak_brier": float(np.square(probability - labels).mean()),
                **classification(labels, probability, cutoff),
            }
        )
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=ROOT / "code/data/raw/operational_2026/commune_monthly_thresholds.csv",
    )
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/moh_hybrid_tuning_ensemble"
    )
    parser.add_argument("--weight-units", type=int, default=10)
    parser.add_argument("--minimum-precision", type=float, default=0.50)
    parser.add_argument("--minimum-hybrid-weight", type=float, default=0.10)
    parser.add_argument(
        "--promote-current",
        action="store_true",
        help="Replace the existing proposed-hybrid rows in results/current.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    moh = load_moh(args.thresholds)

    hybrid_dir = args.artifacts / "DEEPST-INLA"
    test_frame, test_samples = load_npz_split(hybrid_dir, "test")
    valid_frame, valid_samples = load_lognormal_validation(hybrid_dir)
    test_frame = attach_moh(test_frame, moh)
    valid_frame = attach_moh(valid_frame, moh)

    member_specs = [
        ("DeepST-INLA", None),
        ("TCN-GNN", "TCN-GNN-BASE"),
        ("DeepST-Alert", "DEEPST-ALERT"),
        ("LSTM-GNN", "LSTM-GNN"),
    ]
    valid_members = {"DeepST-INLA": valid_samples}
    test_members = {"DeepST-INLA": test_samples}
    for label, relative in member_specs[1:]:
        member_valid_frame, member_valid = load_npz_split(
            args.artifacts / relative, "validation"
        )
        member_test_frame, member_test = load_npz_split(args.artifacts / relative, "test")
        valid_members[label] = align(valid_frame, member_valid_frame, member_valid)
        test_members[label] = align(test_frame, member_test_frame, member_test)

    tuned_policies = {}
    stack_policies = {}
    diagnostics = {}
    tuned_valid_probability = {}
    stack_valid_probability = {}
    tuned_test_probability = {}
    stack_test_probability = {}
    names = [item[0] for item in member_specs]
    for horizon in sorted(valid_samples):
        valid_part = valid_frame[valid_frame["horizon"].eq(horizon)].reset_index(drop=True)
        test_part = test_frame[test_frame["horizon"].eq(horizon)].reset_index(drop=True)
        valid_threshold = valid_part["outbreak_threshold"].to_numpy(float)
        test_threshold = test_part["outbreak_threshold"].to_numpy(float)
        labels = valid_part["target_cases"].to_numpy(float) > valid_threshold
        valid_probability_members = [
            (valid_members[name][horizon] > valid_threshold[:, None]).mean(axis=1)
            for name in names
        ]
        test_probability_members = [
            (test_members[name][horizon] > test_threshold[:, None]).mean(axis=1)
            for name in names
        ]
        tuned_policy, tuned_grid = select_policy(
            labels,
            valid_probability_members[:1],
            names[:1],
            weight_units=1,
            minimum_precision=args.minimum_precision,
        )
        stack_policy, stack_grid = select_policy(
            labels,
            valid_probability_members,
            names,
            weight_units=args.weight_units,
            minimum_precision=args.minimum_precision,
            minimum_first_weight=args.minimum_hybrid_weight,
        )
        tuned_policies[horizon] = tuned_policy
        stack_policies[horizon] = stack_policy
        tuned_valid_probability[horizon] = apply_policy(
            valid_probability_members[:1], tuned_policy
        )
        stack_valid_probability[horizon] = apply_policy(
            valid_probability_members, stack_policy
        )
        tuned_test_probability[horizon] = apply_policy(
            test_probability_members[:1], tuned_policy
        )
        stack_test_probability[horizon] = apply_policy(
            test_probability_members, stack_policy
        )
        diagnostics[str(horizon)] = {
            "tuned_hybrid_candidates": len(tuned_grid),
            "deep_stack_candidates": len(stack_grid),
            "tuned_hybrid": tuned_policy,
            "deep_stack": stack_policy,
        }

    candidates = [
        (
            "DeepST-INLA (MOH F1-tuned)",
            tuned_policies,
            tuned_valid_probability,
            tuned_test_probability,
        ),
        (
            "DeepST-INLA + deep stack (MOH)",
            stack_policies,
            stack_valid_probability,
            stack_test_probability,
        ),
    ]
    full_rows = []
    later_rows = []
    for label, policies, valid_probability, test_probability in candidates:
        valid_metrics, valid_predictions = score(
            label,
            valid_frame,
            valid_samples,
            valid_probability,
            policies,
            "2019-01-01",
            "2020-12-01",
            "validation",
        )
        full_metrics, predictions = score(
            label,
            test_frame,
            test_samples,
            test_probability,
            policies,
            "2021-01-01",
            "2025-12-01",
            "test",
        )
        later_metrics, _ = score(
            label,
            test_frame,
            test_samples,
            test_probability,
            policies,
            "2023-01-01",
            "2025-12-01",
            "test_2023_2025",
        )
        slug = "tuned_hybrid" if "F1-tuned" in label else "hybrid_deep_stack"
        directory = args.output / slug
        directory.mkdir(parents=True, exist_ok=True)
        valid_metrics.to_csv(directory / "validation_metrics.csv", index=False)
        valid_predictions.to_csv(directory / "validation_predictions.csv", index=False)
        full_metrics.to_csv(directory / "metrics.csv", index=False)
        predictions.to_csv(directory / "predictions.csv", index=False)
        later_metrics.to_csv(directory / "metrics_2023_2025.csv", index=False)
        full_rows.append(full_metrics)
        later_rows.append(later_metrics)

    candidate_full = pd.concat(full_rows, ignore_index=True)
    candidate_later = pd.concat(later_rows, ignore_index=True)
    candidate_full.to_csv(args.output / "candidate_metrics_by_horizon.csv", index=False)
    summarize(candidate_full).to_csv(
        args.output / "candidate_metrics_mean_H1_H3.csv", index=False
    )
    candidate_later.to_csv(
        args.output / "candidate_metrics_by_horizon_2023_2025.csv", index=False
    )
    summarize(candidate_later).to_csv(
        args.output / "candidate_metrics_mean_H1_H3_2023_2025.csv", index=False
    )

    current = pd.read_csv(ROOT / "results/current/final_metrics_by_horizon.csv")
    comparison = pd.concat([current, candidate_full], ignore_index=True)
    comparison.to_csv(args.output / "comparison_metrics_by_horizon.csv", index=False)
    summarize(comparison).to_csv(
        args.output / "comparison_metrics_mean_H1_H3.csv", index=False
    )
    current_later = pd.read_csv(
        ROOT / "results/current/final_metrics_by_horizon_2023_2025.csv"
    )
    comparison_later = pd.concat([current_later, candidate_later], ignore_index=True)
    comparison_later.to_csv(
        args.output / "comparison_metrics_by_horizon_2023_2025.csv", index=False
    )
    summarize(comparison_later).to_csv(
        args.output / "comparison_metrics_mean_H1_H3_2023_2025.csv", index=False
    )
    (args.output / "selection.json").write_text(
        json.dumps(
            {
                "selection_split": "validation 2019-01 through 2020-12 only",
                "locked_test_split": "2021-01 through 2025-12",
                "threshold_definition": "audited MOH",
                "objective": "maximum validation F1 subject to minimum precision",
                "minimum_precision": args.minimum_precision,
                "weight_grid_step": 1 / args.weight_units,
                "minimum_hybrid_weight_in_stack": args.minimum_hybrid_weight,
                "count_distribution": "unchanged retained DeepST-INLA distribution",
                "test_outcomes_used_for_selection": False,
                "diagnostics": diagnostics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.promote_current:
        tuned_label = "DeepST-INLA (MOH F1-tuned)"
        current_label = "DeepST-INLA (proposed hybrid)"
        promoted_full = candidate_full[candidate_full["model"].eq(tuned_label)].copy()
        promoted_later = candidate_later[
            candidate_later["model"].eq(tuned_label)
        ].copy()
        promoted_full["model"] = current_label
        promoted_later["model"] = current_label
        current_full = current[~current["model"].eq(current_label)]
        current_later = current_later[~current_later["model"].eq(current_label)]
        published_full = pd.concat(
            [current_full, promoted_full], ignore_index=True
        ).sort_values(["model", "horizon"])
        published_later = pd.concat(
            [current_later, promoted_later], ignore_index=True
        ).sort_values(["model", "horizon"])
        current_root = ROOT / "results/current"
        published_full.to_csv(
            current_root / "final_metrics_by_horizon.csv", index=False
        )
        summarize(published_full).to_csv(
            current_root / "final_metrics_mean_H1_H3.csv", index=False
        )
        published_later.to_csv(
            current_root / "final_metrics_by_horizon_2023_2025.csv", index=False
        )
        summarize(published_later).to_csv(
            current_root / "final_metrics_mean_H1_H3_2023_2025.csv", index=False
        )
        provenance_path = current_root / "moh_rescoring_provenance.json"
        provenance = (
            json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance_path.exists()
            else {}
        )
        provenance["promoted_hybrid_alert_policy"] = {
            "selection_split": "2019-01 through 2020-12 only",
            "objective": "maximum validation F1 subject to precision >= 0.50",
            "test_outcomes_used_for_selection": False,
            "count_distribution_changed": False,
            "policies": {
                str(horizon): policy for horizon, policy in tuned_policies.items()
            },
        }
        provenance_path.write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
    print(summarize(comparison).to_string(index=False))


if __name__ == "__main__":
    main()
