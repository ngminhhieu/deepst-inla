#!/usr/bin/env python3
"""Attach unchanged external covariates to a core-feature panel updated from MongoDB."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_open_covariates import mosquito_suitability, radiation_import_pressure


PRESERVED = [
    "nino34_lag2", "iod_dmi_lag1", "soil_moisture", "solar_downward",
    "dewpoint_c", "et0_mm", "elevation_m", "neighbor_dengue_log",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updated-core", type=Path, required=True)
    parser.add_argument("--previous-panel", type=Path, required=True)
    parser.add_argument("--centroids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    core = pd.read_csv(args.updated_core, dtype={"fcode": str, "l1_code": str})
    previous = pd.read_csv(args.previous_panel, dtype={"fcode": str})
    keys = ["fcode", "year", "month"]
    preserved = previous[keys + PRESERVED]
    output = core.merge(preserved, on=keys, how="left", validate="one_to_one")
    if output[PRESERVED].isna().any().any():
        missing = output.loc[output[PRESERVED].isna().any(axis=1), keys].head()
        raise ValueError(f"Missing preserved external covariates:\n{missing}")

    output["date"] = pd.to_datetime(
        {"year": output.year, "month": output.month, "day": 1}
    )
    output["mosquito_suitability"] = mosquito_suitability(
        output["t2m_avg"].to_numpy(float)
    )
    centroids = pd.read_csv(args.centroids, dtype={"fcode": str})
    radiation = radiation_import_pressure(output, centroids)
    output = output.merge(radiation, on=["fcode", "date"], how="left", validate="one_to_one")
    output = output.drop(columns="date")

    original_order = list(previous.columns)
    missing_columns = set(original_order) - set(output.columns)
    if missing_columns:
        raise ValueError(f"Updated panel is missing columns: {sorted(missing_columns)}")
    output = output[original_order].sort_values(["fcode", "year", "month"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"rows={len(output)} communes={output.fcode.nunique()} columns={len(output.columns)}")


if __name__ == "__main__":
    main()
