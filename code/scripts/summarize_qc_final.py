#!/usr/bin/env python3
"""Validate and summarize final QC predictions without mixing legacy outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = {
    "INLA-TS": Path("inla_test_exact/INLA-TS"),
    "Standalone GNN": Path("REPR-GNN"),
    "Representation-INLA": Path(
        "inla_gnn_representation_final/test/INLA-TS-GNN-RES-P4"
    ),
    "Proposal": Path(
        "proposal_tuned/PROPOSAL-TUNED"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    args = parser.parse_args()
    root = args.results_root
    exclusions = pd.read_csv(
        root / "data_quality/excluded_communes.csv", dtype={"fcode": str}
    )
    excluded = set(exclusions["fcode"])
    metric_rows = []
    prediction_manifest = {}
    identity = None
    for display_name, relative in MODELS.items():
        directory = root / relative
        predictions = pd.read_csv(directory / "predictions.csv", dtype={"fcode": str})
        leaked = sorted(excluded.intersection(predictions["fcode"]))
        if leaked:
            raise ValueError(f"Excluded communes leaked into {display_name}: {leaked}")
        identity_columns = [
            "fcode", "issue_date", "target_date", "horizon", "target_cases", "population"
        ]
        current_identity = predictions[identity_columns].sort_values(
            ["fcode", "issue_date", "target_date", "horizon"]
        ).reset_index(drop=True)
        if identity is None:
            identity = current_identity
        else:
            keys = ["fcode", "issue_date", "target_date", "horizon"]
            aligned = identity[keys].equals(current_identity[keys])
            values_match = all(
                np.allclose(identity[column], current_identity[column])
                for column in ("target_cases", "population")
            )
            if not aligned or not values_match:
                raise ValueError(f"Prediction rows are not aligned for {display_name}")
        metrics = pd.read_csv(directory / "metrics.csv")
        metrics.insert(0, "display_model", display_name)
        metric_rows.append(metrics)
        h3 = metrics[metrics["horizon"].eq(3)].copy()
        if len(h3) != 1:
            raise ValueError(f"Expected one H3 metric row for {display_name}")
        prediction_manifest[display_name] = str((relative / "predictions.csv"))

    all_metrics = pd.concat(metric_rows, ignore_index=True)
    all_metrics.to_csv(root / "final_metrics.csv", index=False)
    summary = all_metrics[all_metrics["horizon"].eq(3)].reset_index(drop=True)
    summary.to_csv(root / "final_h3_metrics.csv", index=False)
    comparisons = []
    directions = {
        "mae": "lower",
        "rmse": "lower",
        "crps": "lower",
        "outbreak_brier": "lower",
        "outbreak_precision": "higher",
        "outbreak_specificity": "higher",
        "outbreak_sensitivity": "higher",
        "outbreak_f1": "higher",
    }
    for horizon in sorted(all_metrics["horizon"].unique()):
        part = all_metrics[all_metrics["horizon"].eq(horizon)].set_index(
            "display_model"
        )
        for metric, direction in directions.items():
            baseline = float(part.loc["INLA-TS", metric])
            proposal = float(part.loc["Proposal", metric])
            improved = proposal < baseline if direction == "lower" else proposal > baseline
            comparisons.append(
                {
                    "horizon": int(horizon),
                    "metric": metric,
                    "direction": direction,
                    "inla_ts": baseline,
                    "proposal": proposal,
                    "relative_change_percent": 100 * (proposal - baseline) / baseline,
                    "proposal_improved": improved,
                }
            )
    comparison = pd.DataFrame(comparisons)
    # This is an ablation report, not a hard requirement that the proposal win
    # every metric. In particular, precision and specificity measure different
    # error trade-offs and can legitimately move in opposite directions.
    comparison.to_csv(root / "proposal_vs_inla.csv", index=False)
    failed = comparison.loc[~comparison["proposal_improved"], ["horizon", "metric"]]
    if not failed.empty:
        print("Metrics where the proposal does not improve over INLA-TS:")
        print(failed.to_string(index=False))
    manifest = {
        "results_version": "qc_v1",
        "data_quality_manifest": "data_quality/manifest.json",
        "excluded_fcodes": sorted(excluded),
        "communes": int(identity["fcode"].nunique()),
        "prediction_rows_per_model": len(identity),
        "horizons": sorted(int(value) for value in identity["horizon"].unique()),
        "models": prediction_manifest,
        "proposal_selection": "proposal_tuned/PROPOSAL-TUNED/selection.json",
        "final_metrics": "final_metrics.csv",
        "final_h3_metrics": "final_h3_metrics.csv",
        "proposal_vs_inla": "proposal_vs_inla.csv",
    }
    (root / "final_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary[[
        "display_model", "mae", "rmse", "crps", "outbreak_precision",
        "outbreak_specificity", "outbreak_sensitivity", "outbreak_f1",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
