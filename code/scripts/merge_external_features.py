#!/usr/bin/env python3
"""Merge vetted external covariates into the commune-month panel.

Supported levels/frequencies:
  commune-static   keys: fcode
  commune-monthly  keys: fcode + (date or year/month)
  province-static  keys: l1_code
  province-monthly keys: l1_code + (date or year/month)

This script deliberately does not download data or invent values. It makes the
join explicit so sources such as WorldPop, census infrastructure, serotype,
vector surveillance, satellite covariates, or health-access data can be audited.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def normalise_month(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "date" in frame.columns:
        date = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()
        frame["year"] = date.dt.year
        frame["month"] = date.dt.month
        frame = frame.drop(columns=["date"])
    if not {"year", "month"}.issubset(frame.columns):
        raise ValueError("Monthly feature table needs date or year/month")
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["month"] = pd.to_numeric(frame["month"], errors="raise").astype(int)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monthly", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--level", choices=["commune-static", "commune-monthly", "province-static", "province-monthly"], required=True)
    parser.add_argument("--columns", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = pd.read_csv(args.monthly, dtype={"fcode": str, "l1_code": str})
    features = pd.read_csv(args.features, dtype={"fcode": str, "l1_code": str})
    if args.level.startswith("commune"):
        entity = "fcode"
    else:
        entity = "l1_code"
    monthly = args.level.endswith("monthly")
    keys = [entity]
    if monthly:
        features = normalise_month(features)
        keys += ["year", "month"]
    missing = [column for column in [*keys, *args.columns] if column not in features.columns]
    if missing:
        raise ValueError(f"Feature table missing columns: {missing}")
    if features.duplicated(keys).any():
        raise ValueError(f"Feature table has duplicate keys: {keys}")
    collision = sorted(set(args.columns).intersection(base.columns))
    if collision:
        raise ValueError(f"Refusing to overwrite existing columns: {collision}")
    merged = base.merge(features[[*keys, *args.columns]], on=keys, how="left", validate="many_to_one")
    if merged[args.columns].isna().any().any():
        bad = merged.loc[merged[args.columns].isna().any(axis=1), keys].head(10).to_dict("records")
        raise ValueError(f"External features do not cover the full panel; examples: {bad}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"wrote {args.output}; merged {args.columns}")


if __name__ == "__main__":
    main()
