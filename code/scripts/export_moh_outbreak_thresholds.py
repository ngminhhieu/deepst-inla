#!/usr/bin/env python3
"""Export auditable monthly outbreak thresholds using dengue_forecast_v1 moh.py."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
V1_ROOT = ROOT.parent / "dengue_forecast_v1"
if str(V1_ROOT) not in sys.path:
    sys.path.insert(0, str(V1_ROOT))

from dengue_training.common.moh import build_moh_thresholds  # noqa: E402
from dengue_training.common.settings import PipelineSettings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=V1_ROOT / "dengue_training/data/processed/monthly_commune_from_mongodb.csv",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=V1_ROOT / "dengue_training/config.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "code/data/raw/operational_2026/commune_monthly_thresholds.csv",
    )
    parser.add_argument("--first-year", type=int, default=2015)
    parser.add_argument("--last-year", type=int, default=2026)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = PipelineSettings.from_config(config)
    source = pd.read_csv(
        args.input,
        dtype={"fcode": str},
        usecols=["fcode", "year", "month", "obs_dengue_cases"],
    )
    source["date"] = pd.to_datetime(
        {"year": source["year"], "month": source["month"], "day": 1}
    )
    target_dates = pd.date_range(
        f"{args.first_year}-01-01", f"{args.last_year}-12-01", freq="MS"
    )
    thresholds = build_moh_thresholds(
        source[["fcode", "date", "obs_dengue_cases"]],
        target_dates,
        settings,
    )
    output = thresholds.rename(
        columns={
            "date": "target_date",
            "moh_reference_n": "reference_n",
            "moh_reference_years": "reference_years",
            "moh_excluded_years": "excluded_years",
            "moh_reference_mean_cases": "reference_mean_cases",
            "moh_reference_sd_cases": "reference_sd_cases",
            "moh_threshold_2sd_cases": "epi_threshold",
        }
    )
    output["year"] = output["target_date"].dt.year
    output["month"] = output["target_date"].dt.month
    output["threshold_valid"] = output["epi_threshold"].notna()
    output["method"] = "moh_monthly_mean_plus_2sd_exclude_outbreak_year"
    output = output[
        [
            "fcode",
            "year",
            "month",
            "target_date",
            "epi_threshold",
            "threshold_valid",
            "reference_n",
            "reference_years",
            "excluded_years",
            "reference_mean_cases",
            "reference_sd_cases",
            "method",
        ]
    ].sort_values(["fcode", "year", "month"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    print(
        f"wrote {args.output}: rows={len(output)}, units={output.fcode.nunique()}, "
        f"valid={int(output.threshold_valid.sum())}, "
        f"missing={int((~output.threshold_valid).sum())}"
    )


if __name__ == "__main__":
    main()
