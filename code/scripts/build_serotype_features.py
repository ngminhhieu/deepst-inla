#!/usr/bin/env python3
"""Aggregate dated laboratory/genomic dengue serotype records to province-month shares.

Input must be a *sample-level* table with date, province code and serotype. The
script never treats sequence counts as dengue case counts; it only derives the
composition among observed typed samples. A trailing window can stabilize sparse
surveillance while using only information available up to each month.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEROTYPES = (1, 2, 3, 4)


def normalise_serotype(value: object) -> int | None:
    text = str(value).strip().upper().replace("DENV-", "").replace("DENV", "")
    try:
        number = int(float(text))
    except (TypeError, ValueError):
        return None
    return number if number in SEROTYPES else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--province-column", default="l1_code")
    parser.add_argument("--serotype-column", default="serotype")
    parser.add_argument("--window-months", type=int, default=3)
    parser.add_argument("--minimum-typed-samples", type=int, default=3)
    args = parser.parse_args()
    if args.window_months < 1 or args.minimum_typed_samples < 1:
        raise ValueError("window-months and minimum-typed-samples must be >= 1")

    frame = pd.read_csv(args.samples, dtype={args.province_column: str})
    required = {args.date_column, args.province_column, args.serotype_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Serotype samples missing columns: {missing}")
    frame = frame.rename(
        columns={
            args.date_column: "date",
            args.province_column: "l1_code",
            args.serotype_column: "serotype",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    frame["serotype"] = frame["serotype"].map(normalise_serotype)
    frame = frame.dropna(subset=["date", "l1_code", "serotype"]).copy()
    frame["serotype"] = frame["serotype"].astype(int)
    if frame.empty:
        raise ValueError("No valid typed samples remain after parsing")

    monthly = (
        frame.groupby(["l1_code", "date", "serotype"]).size().unstack(fill_value=0)
        .reindex(columns=SEROTYPES, fill_value=0)
    )
    # Create a complete monthly index within each province before rolling so a
    # quiet month is represented by zero new typed samples rather than skipped.
    outputs = []
    for province, part in monthly.groupby(level=0):
        part = part.droplevel(0)
        dates = pd.date_range(part.index.min(), part.index.max(), freq="MS")
        part = part.reindex(dates, fill_value=0)
        counts = part.rolling(args.window_months, min_periods=1).sum()
        total = counts.sum(axis=1)
        shares = counts.div(total.replace(0, np.nan), axis=0)
        shares.columns = [f"denv{number}_share" for number in SEROTYPES]
        result = shares.reset_index(names="date")
        result.insert(0, "l1_code", str(province))
        result["typed_samples_window"] = total.to_numpy()
        result["serotype_feature_available"] = (
            result["typed_samples_window"] >= args.minimum_typed_samples
        ).astype(int)
        # Encode "no reliable typed composition" as zero shares plus an explicit
        # availability flag; this avoids pretending that zeros are true serotype
        # proportions while keeping the downstream panel numeric and complete.
        for column in [f"denv{number}_share" for number in SEROTYPES]:
            result.loc[result["serotype_feature_available"].eq(0), column] = 0.0
        outputs.append(result)

    output = pd.concat(outputs, ignore_index=True)
    output["year"] = output["date"].dt.year
    output["month"] = output["date"].dt.month
    output = output.drop(columns="date")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"wrote {args.output}; {len(output)} province-month rows")


if __name__ == "__main__":
    main()
