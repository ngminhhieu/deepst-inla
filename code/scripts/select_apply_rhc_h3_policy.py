#!/usr/bin/env python3
"""Select the DeepST-INLA H3 policy on RHC validation and apply to 2026."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_multiobjective_ensemble import (
    aligned_column, apply_distribution_policy, apply_platt,
    choose_alert_head, choose_distribution,
)
from exact_inla_deep_ensemble import load_exact_horizon


KEYS = ["fcode", "issue_date", "target_date", "horizon"]


def align_samples(template: pd.DataFrame, source: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    indexed = source.copy()
    indexed["_row"] = np.arange(len(indexed))
    merged = template[KEYS].merge(indexed[KEYS + ["_row"]], on=KEYS, how="left", validate="one_to_one")
    if merged._row.isna().any():
        raise ValueError("Could not align H3 component")
    return values[merged._row.astype(int).to_numpy()]


def validation_component(directory: Path, horizon_index: int = 2) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(directory / "validation_predictions.csv", dtype={"fcode": str})
    cube = np.load(directory / "validation_predictive_samples.npz")["samples"]
    part = frame.loc[frame.horizon.eq(3)].reset_index(drop=True)
    values = cube[:, :, horizon_index, :].reshape(-1, cube.shape[-1])
    return part, values


def test_component(directory: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(directory / "predictions_h3.csv", dtype={"fcode": str})
    values = np.load(directory / "predictive_samples_h3.npz")["samples"]
    return frame, values


def probabilities(part, distribution, deep, peak, peak_frame):
    threshold = part.outbreak_threshold.to_numpy(float)[:, None]
    return [
        (distribution > threshold).mean(axis=1),
        (deep > threshold).mean(axis=1),
        (peak > threshold).mean(axis=1),
        aligned_column(part, peak_frame, "outbreak_probability_auxiliary"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--peak", type=Path, required=True)
    parser.add_argument("--hybrid-validation", type=Path, required=True)
    parser.add_argument("--hybrid-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=500)
    args = parser.parse_args()

    base_v, base_values_v = validation_component(args.base)
    peak_v, peak_raw_v = validation_component(args.peak)
    peak_values_v = align_samples(base_v, peak_v, peak_raw_v)
    hybrid_values_v = load_exact_horizon(args.hybrid_validation, base_v, 3, args.draws)
    peak_risk_v = aligned_column(base_v, peak_v, "outbreak_probability")
    distribution_policy, distribution_grid = choose_distribution(
        base_v, base_values_v, peak_values_v, hybrid_values_v, peak_risk_v, 100
    )
    selected_v = apply_distribution_policy(
        distribution_policy, base_values_v, peak_values_v, hybrid_values_v, peak_risk_v, args.draws
    )
    alert_policy, alert_grid = choose_alert_head(
        base_v, probabilities(base_v, selected_v, base_values_v, peak_values_v, peak_v)
    )

    base_t, base_values_t = test_component(args.base)
    peak_t, peak_raw_t = test_component(args.peak)
    peak_values_t = align_samples(base_t, peak_t, peak_raw_t)
    hybrid_values_t = load_exact_horizon(args.hybrid_test, base_t, 3, args.draws)
    peak_risk_t = aligned_column(base_t, peak_t, "outbreak_probability")
    selected_t = apply_distribution_policy(
        distribution_policy, base_values_t, peak_values_t, hybrid_values_t, peak_risk_t, args.draws
    )
    members = probabilities(base_t, selected_t, base_values_t, peak_values_t, peak_t)
    raw_probability = sum(
        float(weight) * member for weight, member in zip(alert_policy["weights"], members)
    )
    alert_probability = apply_platt(
        raw_probability, float(alert_policy["platt_intercept"]), float(alert_policy["platt_slope"])
    )
    output = base_t.copy()
    output["model"] = "DEEPST-INLA-RHC-H3"
    output["predicted_mean"] = selected_t.mean(axis=1)
    output["predicted_median"] = np.median(selected_t, axis=1)
    output["outbreak_probability"] = alert_probability
    output["outbreak_decision_threshold"] = float(alert_policy["decision_threshold"])
    output["distribution_moh_probability"] = members[0]
    args.output.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output / "predictions_h3.csv", index=False)
    np.savez_compressed(args.output / "predictive_samples_h3.npz", samples=selected_t)
    selection = {
        "selection_split": "RHC validation 2019-2020", "horizon": 3,
        "distribution": distribution_policy, "alert": alert_policy,
        "distribution_candidates": len(distribution_grid), "alert_candidates": len(alert_grid),
    }
    (args.output / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection))


if __name__ == "__main__":
    main()
