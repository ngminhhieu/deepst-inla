#!/usr/bin/env python3
"""Convert origin-destination mobility flows into a monthly import-pressure feature.

Expected flows: date (or year/month), origin_fcode, destination_fcode, flow.
For destination i at month t:
  pressure_i,t = sum_j flow_j->i,t * log1p(incidence_j,t) / sum_j flow_j->i,t

Only same-month or earlier case information is used. If reporting delay is a concern,
use --case-lag 1 (recommended sensitivity analysis) so pressure at t uses cases t-1.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def month_start(frame: pd.DataFrame) -> pd.Series:
    if "date" in frame.columns:
        return pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()
    return pd.to_datetime({"year": frame["year"], "month": frame["month"], "day": 1})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monthly", type=Path, required=True)
    parser.add_argument("--flows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-lag", type=int, default=0)
    parser.add_argument("--feature-name", default="mobility_import_pressure")
    args = parser.parse_args()
    if args.case_lag < 0:
        raise ValueError("case-lag must be >= 0")

    monthly = pd.read_csv(args.monthly, dtype={"fcode": str})
    monthly["date"] = month_start(monthly)
    monthly["incidence"] = monthly["obs_dengue_cases"] / monthly["population"] * 100_000.0
    source = monthly[["fcode", "date", "incidence"]].rename(columns={"fcode": "origin_fcode"})
    if args.case_lag:
        source["date"] = source["date"] + pd.offsets.MonthBegin(args.case_lag)

    flows = pd.read_csv(args.flows, dtype={"origin_fcode": str, "destination_fcode": str})
    flows["date"] = month_start(flows)
    required = {"origin_fcode", "destination_fcode", "flow", "date"}
    missing = sorted(required.difference(flows.columns))
    if missing:
        raise ValueError(f"Mobility file missing columns: {missing}")
    flows["flow"] = pd.to_numeric(flows["flow"], errors="coerce")
    if flows["flow"].isna().any() or (flows["flow"] < 0).any():
        raise ValueError("flow must be non-negative numeric")
    flows = flows.merge(source, on=["origin_fcode", "date"], how="left", validate="many_to_one")
    if flows["incidence"].isna().any():
        raise ValueError("Some mobility origins/months do not match the dengue panel")
    flows["weighted_risk"] = flows["flow"] * np.log1p(flows["incidence"])
    grouped = flows.groupby(["destination_fcode", "date"], as_index=False).agg(
        weighted_risk=("weighted_risk", "sum"), total_flow=("flow", "sum")
    )
    grouped[args.feature_name] = grouped["weighted_risk"] / grouped["total_flow"].replace(0, np.nan)
    grouped[args.feature_name] = grouped[args.feature_name].fillna(0.0)
    grouped = grouped.rename(columns={"destination_fcode": "fcode"})[["fcode", "date", args.feature_name]]

    output = monthly.drop(columns=["incidence"]).merge(grouped, on=["fcode", "date"], how="left", validate="one_to_one")
    output[args.feature_name] = output[args.feature_name].fillna(0.0)
    output = output.drop(columns=["date"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"wrote {args.output} with {args.feature_name}")


if __name__ == "__main__":
    main()
