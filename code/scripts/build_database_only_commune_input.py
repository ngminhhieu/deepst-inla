#!/usr/bin/env python3
"""Build a core commune panel using only MongoDB-exported fields.

Historical rows come from the project's direct MongoDB CSV export.  Rows for
2026 are replaced by the newer read-only MongoDB JSON snapshot so revisions in
cases, population and climate are retained.  No engineered or public/external
covariates are added.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


COLUMNS = [
    "fcode", "year", "month", "obs_dengue_cases", "population", "area",
    "l1_code", "ws_avg", "ws_min", "ws_max", "t2m_avg", "t2m_min",
    "t2m_max", "rh_avg", "rh_min", "rh_max", "tp_accum",
]
CLIMATE = COLUMNS[7:]


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return f"external:{resolved.name}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric(value):
    try:
        number = float(value)
        return number if number > 0 else None
    except (TypeError, ValueError):
        return None


def resolve_population(boundary_value, document_value, year: int):
    for value in (boundary_value, document_value):
        if isinstance(value, dict):
            usable = {
                str(key): numeric(item) for key, item in value.items()
                if numeric(item) is not None
            }
            if str(year) in usable:
                return usable[str(year)]
            if usable:
                latest = max(usable, key=lambda key: int(key) if key.isdigit() else -1)
                return usable[latest]
        else:
            number = numeric(value)
            if number is not None:
                return number
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-mongo-csv", type=Path, required=True)
    parser.add_argument("--current-mongo-json", type=Path, required=True)
    parser.add_argument("--model-universe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.model_universe.open(newline="", encoding="utf-8-sig") as handle:
        units = {row["fcode"] for row in csv.DictReader(handle)}
    rows = {}
    with args.historical_mongo_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (row["fcode"], int(row["year"]), int(row["month"]))
            if key[0] in units and (key[1] < 2026 or (key[1] == 2026 and key[2] <= 8)):
                rows[key] = {column: row[column] for column in COLUMNS}

    payload = json.loads(args.current_mongo_json.read_text(encoding="utf-8"))
    dengue = {(str(row["fcode"]), int(row["year"])): row for row in payload["dengue"]}
    climate = {(str(row["fcode"]), int(row["year"])): row for row in payload["climate"]}
    boundary = {str(row["fcode"]): row for row in payload["boundary"]}
    for fcode in sorted(units, key=int):
        dengue_doc = dengue[(fcode, 2026)]
        climate_doc = climate[(fcode, 2026)]
        boundary_props = boundary[fcode].get("properties") or {}
        document_props = dengue_doc.get("properties") or {}
        for month in range(1, 9):
            dengue_month = (dengue_doc.get("months") or {})[str(month)]
            climate_month = (climate_doc.get("months") or {})[str(month)]
            rows[(fcode, 2026, month)] = {
                "fcode": fcode,
                "year": 2026,
                "month": month,
                "obs_dengue_cases": dengue_month["obs_dengue_cases"],
                "population": resolve_population(
                    boundary_props.get("population"), document_props.get("population"), 2026
                ),
                "area": boundary_props.get("area", document_props.get("area")),
                "l1_code": boundary_props.get("l1_code", document_props.get("l1_code")),
                **{field: climate_month[field] for field in CLIMATE},
            }

    expected = len(units) * 200  # Jan 2010 through Aug 2026 inclusive.
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} commune-months, found {len(rows)}")
    expected_keys = {
        (fcode, year, month)
        for fcode in units
        for year in range(2010, 2027)
        for month in range(1, 13)
        if year < 2026 or month <= 8
    }
    missing = expected_keys - set(rows)
    if missing:
        raise ValueError(f"Missing commune-months: {sorted(missing)[:5]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for key in sorted(rows, key=lambda item: (int(item[0]), item[1], item[2])):
            writer.writerow(rows[key])
    manifest = {
        "rows": len(rows),
        "communes": len(units),
        "date_start": "2010-01-01",
        "date_end": "2026-08-31",
        "historical_source": portable_path(args.historical_mongo_csv),
        "historical_source_sha256": sha256(args.historical_mongo_csv),
        "current_2026_source": portable_path(args.current_mongo_json),
        "current_2026_source_sha256": sha256(args.current_mongo_json),
        "current_2026_override_months": [f"2026-{month:02d}" for month in range(1, 9)],
        "scope": "554 model communes; QC-excluded commune 31495 is not included",
        "fields": "MongoDB surveillance, population/boundary and core climate only",
        "engineered_or_external_covariates": False,
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
