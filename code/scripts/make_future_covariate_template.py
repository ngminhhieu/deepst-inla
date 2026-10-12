#!/usr/bin/env python3
"""Create the exact issue-date/target-date schema expected for future forecasts."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monthly", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--columns", nargs="+", default=["t2m_forecast", "rh_forecast", "tp_forecast"])
    args = parser.parse_args()
    frame = pd.read_csv(args.monthly, dtype={"fcode": str})
    dates = pd.to_datetime({"year": frame["year"], "month": frame["month"], "day": 1})
    start, end = dates.min(), dates.max()
    units = sorted(frame["fcode"].unique())
    issue_dates = pd.date_range(start, end, freq="MS")
    rows = []
    for issue in issue_dates:
        for horizon in args.horizons:
            target = issue + pd.offsets.MonthBegin(horizon)
            if target > end:
                continue
            for unit in units:
                rows.append({"fcode": unit, "issue_date": issue, "target_date": target, **{column: pd.NA for column in args.columns}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {args.output}; fill only forecasts that were available at each issue_date")


if __name__ == "__main__":
    main()
