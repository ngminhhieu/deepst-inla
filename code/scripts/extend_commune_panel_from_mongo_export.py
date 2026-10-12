#!/usr/bin/env python3
"""Extend the locked commune panel with observed 2026 MongoDB months.

The Mongo export is read-only input.  Surveillance and core climate fields are
updated from MongoDB; static fields and public covariates follow the paper's
causal forward-fill rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CORE_CLIMATE = [
    "ws_avg", "ws_min", "ws_max", "t2m_avg", "t2m_min", "t2m_max",
    "rh_avg", "rh_min", "rh_max", "tp_accum",
]


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return f"external:{resolved.name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mongo-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--months", type=int, nargs="+", default=[7, 8])
    args = parser.parse_args()

    source = pd.read_csv(args.source, dtype={"fcode": str, "l1_code": str})
    payload = json.loads(args.mongo_export.read_text(encoding="utf-8"))
    dengue = {str(row["fcode"]): row for row in payload["dengue"]}
    climate = {str(row["fcode"]): row for row in payload["climate"]}
    units = sorted(source["fcode"].unique())
    latest = (
        source.sort_values(["year", "month"]).groupby("fcode", as_index=False).tail(1)
        .set_index("fcode")
    )

    additions = []
    for month in sorted(set(args.months)):
        for fcode in units:
            if fcode not in dengue or fcode not in climate:
                raise ValueError(f"Mongo export does not cover commune {fcode}")
            dmonth = dengue[fcode].get("months", {}).get(str(month))
            cmonth = climate[fcode].get("months", {}).get(str(month))
            if dmonth is None or cmonth is None:
                raise ValueError(f"Mongo export lacks 2026-{month:02d} for {fcode}")
            row = latest.loc[fcode].copy()
            row["year"] = 2026
            row["month"] = month
            row["obs_dengue_cases"] = dmonth["obs_dengue_cases"]
            for column in CORE_CLIMATE:
                row[column] = cmonth[column]
            additions.append({"fcode": fcode, **row.to_dict()})

    output = pd.concat([source, pd.DataFrame(additions)], ignore_index=True)
    output = output.sort_values(["fcode", "year", "month"])
    if output.duplicated(["fcode", "year", "month"]).any():
        raise ValueError("Extended panel contains duplicate commune-month rows")
    expected = len(units) * len(set(args.months))
    if len(additions) != expected:
        raise ValueError(f"Expected {expected} appended rows, got {len(additions)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    manifest = {
        "source": portable_path(args.source),
        "mongo_export": portable_path(args.mongo_export),
        "appended_months": [f"2026-{month:02d}" for month in sorted(set(args.months))],
        "communes": len(units),
        "rows": len(output),
        "public_covariates": "causal forward-fill from the latest available value",
        "surveillance_and_core_climate": "MongoDB 2026 monthly records",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
