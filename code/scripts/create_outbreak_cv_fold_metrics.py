#!/usr/bin/env python3
"""Create target-month rolling CV metrics from validation outbreak flags."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_v1_outbreak_methods import binary_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-alerts", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.validation_alerts, dtype={"fcode": str})
    policies = pd.read_csv(args.policies)
    rows = []
    for (target_date, horizon), part in frame.groupby(["target_date", "horizon"], sort=True):
        observed = part["observed_outbreak"].astype(bool).to_numpy()
        probability = part["outbreak_probability"].to_numpy(float)
        for method, predicted in (
            ("current_probability", part["flag_current_probability"].astype(bool).to_numpy()),
            ("risk_abs", part["flag_risk_abs"].astype(bool).to_numpy()),
            ("risk_z", part["flag_risk_z"].astype(bool).to_numpy()),
            ("pred_mean", part["flag_pred_mean"].astype(bool).to_numpy()),
        ):
            extra = {"brier": float(np.square(probability - observed).mean())} if method == "current_probability" else {"brier": np.nan}
            rows.append({"target_month": target_date, "horizon": int(horizon), "method": method,
                         "incidence_cutoff": np.nan, **binary_metrics(observed, predicted), **extra})
        for method, minimum in (("two_of_three", 2), ("three_of_three", 3)):
            cutoff = int(policies.loc[(policies.horizon.eq(horizon)) & policies.method.eq(method), "incidence_cutoff"].iloc[0])
            predicted = (part["methods_count"].to_numpy(int) >= minimum) & (part["pred_mean_incidence"].to_numpy(float) > cutoff)
            rows.append({"target_month": target_date, "horizon": int(horizon), "method": method,
                         "incidence_cutoff": cutoff, **binary_metrics(observed, predicted), "brier": np.nan})
    folds = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output / "cv_fold_metrics.csv", index=False)
    summary = folds.groupby(["horizon", "method"], as_index=False).agg(
        folds=("target_month", "nunique"), mean_precision=("precision", "mean"),
        mean_specificity=("specificity", "mean"), mean_sensitivity=("sensitivity", "mean"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"), mean_f1=("f1", "mean"),
        mean_brier=("brier", "mean"), sd_f1=("f1", "std"),
    )
    summary.to_csv(args.output / "cv_summary_by_horizon_method.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
