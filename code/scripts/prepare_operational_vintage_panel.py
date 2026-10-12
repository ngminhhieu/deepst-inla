#!/usr/bin/env python3
"""Append target-only rows needed for one operational H1--H3 forecast vintage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return f"external:{resolved.name}"


def month_start(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("M").to_timestamp()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--issue-date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    issue = month_start(args.issue_date)
    frame = pd.read_csv(args.source, dtype={"fcode": str})
    required = {"fcode", "year", "month", "obs_dengue_cases", "population"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Source panel is missing columns: {missing}")
    frame["date"] = pd.to_datetime(
        dict(year=frame["year"], month=frame["month"], day=1)
    )
    frame = frame.loc[frame["date"].le(issue)].copy()
    issue_rows = frame.loc[frame["date"].eq(issue)].sort_values("fcode")
    units = sorted(frame["fcode"].unique())
    if len(issue_rows) != len(units) or issue_rows["fcode"].nunique() != len(units):
        raise ValueError(
            f"Issue month {issue.date()} does not have exactly one row for every unit"
        )

    future_rows = []
    for horizon in sorted(set(args.horizons)):
        target = issue + pd.DateOffset(months=int(horizon))
        part = issue_rows.copy()
        part["year"] = int(target.year)
        part["month"] = int(target.month)
        part["date"] = target
        # Target outcomes are deliberately unavailable.  The forecasting code
        # needs materialised target dates but never uses these values as input.
        part["obs_dengue_cases"] = 0
        future_rows.append(part)

    output = pd.concat([frame, *future_rows], ignore_index=True)
    output = output.sort_values(["fcode", "year", "month"])
    if output.duplicated(["fcode", "year", "month"]).any():
        raise ValueError("Operational panel contains duplicate commune-month rows")
    output = output.drop(columns="date")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = {
        "source": portable_path(args.source),
        "issue_date": str(issue.date()),
        "target_months": [
            str((issue + pd.DateOffset(months=int(h))).date())
            for h in sorted(set(args.horizons))
        ],
        "units": len(units),
        "source_latest_month": str(
            pd.to_datetime(
                dict(year=frame["year"], month=frame["month"], day=1)
            ).max().date()
        ),
        "future_target_outcomes": "unavailable; zero placeholders are never model inputs",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
