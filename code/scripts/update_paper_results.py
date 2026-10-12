#!/usr/bin/env python3
"""Update manuscript macros used by the H3-focused diagnostics."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument("--paper-dir", default="../paper")
    args = parser.parse_args()
    results = Path(args.results_dir).resolve()
    paper = Path(args.paper_dir).resolve()

    metrics = {}
    widths = {}
    peak_summaries = {}
    for model in ("M1", "M3", "M4", "M6", "M6-INLA"):
        metric_path = results / model / "metrics.csv"
        prediction_path = results / model / "predictions.csv"
        if not metric_path.exists() or not prediction_path.exists():
            raise SystemExit(f"Missing locked H3 result for {model}")
        frame = pd.read_csv(metric_path)
        metrics[model] = frame.loc[frame["horizon"] == 3].iloc[0]
        predictions = pd.read_csv(prediction_path)
        predictions = predictions[predictions["horizon"] == 3]
        widths[model] = (
            (predictions["upper_80"] - predictions["lower_80"])
            / predictions["population"] * 100_000
        ).mean()
        if model in {"M4", "M6"}:
            peak_rows = []
            for _, commune in predictions.groupby("fcode"):
                observed = (
                    commune["target_cases"].to_numpy(float)
                    / commune["population"].to_numpy(float) * 100_000
                )
                predicted = (
                    commune["predicted_mean"].to_numpy(float)
                    / commune["population"].to_numpy(float) * 100_000
                )
                correlation = (
                    np.corrcoef(observed, predicted)[0, 1]
                    if np.std(observed) > 0 and np.std(predicted) > 0
                    else np.nan
                )
                peak_rows.append((predicted.max() / observed.max(), correlation))
            peak_summaries[model] = np.nanmedian(np.asarray(peak_rows), axis=0)

    replacements = {
        "MoneCRPS": metrics["M1"]["crps"],
        "MfourCRPS": metrics["M4"]["crps"],
        "MsixCRPS": metrics["M6"]["crps"],
        "HybridCRPS": metrics["M6-INLA"]["crps"],
        "MfourMAE": metrics["M4"]["mae"],
        "MsixMAE": metrics["M6"]["mae"],
        "HybridMAE": metrics["M6-INLA"]["mae"],
        "MfourCoverage": metrics["M4"]["coverage_80"],
        "MsixCoverage": metrics["M6"]["coverage_80"],
        "HybridCoverage": metrics["M6-INLA"]["coverage_80"],
        "MfourBrier": metrics["M4"]["outbreak_brier"],
        "MsixBrier": metrics["M6"]["outbreak_brier"],
        "HybridBrier": metrics["M6-INLA"]["outbreak_brier"],
        "MthreePrecision": metrics["M3"]["outbreak_precision"],
        "MthreeRecall": metrics["M3"]["outbreak_sensitivity"],
        "MthreeFone": metrics["M3"]["outbreak_f1"],
        "MfourPrecision": metrics["M4"]["outbreak_precision"],
        "MfourRecall": metrics["M4"]["outbreak_sensitivity"],
        "MfourFone": metrics["M4"]["outbreak_f1"],
        "MsixPrecision": metrics["M6"]["outbreak_precision"],
        "MsixRecall": metrics["M6"]["outbreak_sensitivity"],
        "MsixFone": metrics["M6"]["outbreak_f1"],
        "HybridPrecision": metrics["M6-INLA"]["outbreak_precision"],
        "HybridRecall": metrics["M6-INLA"]["outbreak_sensitivity"],
        "HybridFone": metrics["M6-INLA"]["outbreak_f1"],
        "MfourPeakRatio": peak_summaries["M4"][0],
        "MfourPeakCorrelation": peak_summaries["M4"][1],
        "MsixPeakRatio": peak_summaries["M6"][0],
        "MsixPeakCorrelation": peak_summaries["M6"][1],
        "MfourWidthEighty": widths["M4"],
        "MsixWidthEighty": widths["M6"],
        "HybridWidthEighty": widths["M6-INLA"],
    }
    path = paper / "results_values.tex"
    text = path.read_text(encoding="utf-8")
    for key, value in replacements.items():
        pattern = rf"\\newcommand\{{\\{key}\}}\{{[^}}]*\}}"
        replacement = rf"\\newcommand{{\\{key}}}{{{float(value):.3f}}}"
        text, count = re.subn(pattern, replacement, text)
        if count != 1:
            raise SystemExit(f"Could not update exactly one macro: {key}")
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
