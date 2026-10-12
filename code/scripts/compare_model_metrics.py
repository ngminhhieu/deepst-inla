#!/usr/bin/env python3
"""Collect comparable probabilistic and outbreak metrics across model runs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPORT_COLUMNS = [
    "model",
    "horizon",
    "mae",
    "rmse",
    "crps",
    "outbreak_brier",
    "coverage_80",
    "coverage_95",
    "outbreak_precision",
    "outbreak_specificity",
    "outbreak_recall",
    "outbreak_sensitivity",
    "outbreak_f1",
    "outbreak_tp",
    "outbreak_fp",
    "outbreak_fn",
    "outbreak_tn",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--models", nargs="*", help="Model subdirectories. Default: every */metrics.csv under results-root")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.results_root
    if args.models:
        metric_paths = [root / model / "metrics.csv" for model in args.models]
    else:
        metric_paths = sorted(root.glob("*/metrics.csv"))
    frames = []
    for path in metric_paths:
        if not path.exists():
            print(f"skip missing {path}")
            continue
        frame = pd.read_csv(path)
        if "model" not in frame.columns:
            frame.insert(0, "model", path.parent.name)
        missing = [column for column in REPORT_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{path} is missing corrected metric columns: {missing}")
        frames.append(frame[REPORT_COLUMNS])
    if not frames:
        raise SystemExit("No compatible metrics.csv files found")
    result = pd.concat(frames, ignore_index=True).sort_values(["horizon", "model"])
    output = args.output or root / "model_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
