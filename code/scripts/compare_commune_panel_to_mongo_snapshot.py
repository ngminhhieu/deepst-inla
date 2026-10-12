#!/usr/bin/env python3
"""Compare the operational commune panel with a read-only Mongo JSON snapshot."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


FIELDS = [
    "obs_dengue_cases", "population", "area", "l1_code",
    "ws_avg", "ws_min", "ws_max", "t2m_avg", "t2m_min", "t2m_max",
    "rh_avg", "rh_min", "rh_max", "tp_accum",
]
CLIMATE = set(FIELDS[4:])


def numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def population(value, year: int):
    if isinstance(value, dict):
        usable = {}
        for key, item in value.items():
            number = numeric(item)
            if number is not None and number > 0:
                usable[str(key)] = number
        if str(year) in usable:
            return usable[str(year)]
        if usable:
            latest = max(usable, key=lambda key: int(key) if key.isdigit() else -1)
            return usable[latest]
        return None
    number = numeric(value)
    return number if number is not None and number > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--mongo-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.panel.open(newline="", encoding="utf-8-sig") as handle:
        panel = {
            (row["fcode"], int(float(row["year"])), int(float(row["month"]))): row
            for row in csv.DictReader(handle)
        }
    payload = json.loads(args.mongo_snapshot.read_text(encoding="utf-8"))
    dengue = {(str(row["fcode"]), int(row["year"])): row for row in payload["dengue"]}
    climate = {(str(row["fcode"]), int(row["year"])): row for row in payload["climate"]}
    boundary = {str(row["fcode"]): row for row in payload["boundary"]}

    mongo = {}
    for key in panel:
        fcode, year, month = key
        dengue_doc = dengue.get((fcode, year))
        climate_doc = climate.get((fcode, year))
        boundary_doc = boundary.get(fcode)
        if not (dengue_doc and climate_doc and boundary_doc):
            continue
        dengue_month = (dengue_doc.get("months") or {}).get(str(month))
        climate_month = (climate_doc.get("months") or {}).get(str(month))
        if not (dengue_month and climate_month):
            continue
        props = boundary_doc.get("properties") or {}
        mongo[key] = {
            "obs_dengue_cases": dengue_month.get("obs_dengue_cases"),
            "population": population(props.get("population"), year),
            "area": props.get("area"),
            "l1_code": props.get("l1_code"),
            **{field: climate_month.get(field) for field in CLIMATE},
        }

    stats = {field: Counter() for field in FIELDS}
    maximum = {field: 0.0 for field in FIELDS}
    differences = []
    for key in sorted(set(panel) & set(mongo)):
        for field in FIELDS:
            left, right = numeric(panel[key].get(field)), numeric(mongo[key].get(field))
            if left is None or right is None:
                stats[field]["missing"] += 1
                continue
            difference = abs(left - right)
            maximum[field] = max(maximum[field], difference)
            stats[field]["compared"] += 1
            if difference == 0:
                stats[field]["exact_match"] += 1
            else:
                stats[field]["nonexact"] += 1
            tolerance = 5e-5 if field in CLIMATE else 0.0
            if difference > tolerance:
                stats[field]["meaningful_mismatch"] += 1
                differences.append({
                    "fcode": key[0], "year": key[1], "month": key[2],
                    "field": field, "panel_value": left, "mongo_value": right,
                    "absolute_difference": difference,
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "commune_panel_vs_mongo_2026_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        columns = [
            "field", "compared", "exact_matches", "nonexact_values",
            "meaningful_mismatches", "max_absolute_difference", "tolerance",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for field in FIELDS:
            writer.writerow({
                "field": field,
                "compared": stats[field]["compared"],
                "exact_matches": stats[field]["exact_match"],
                "nonexact_values": stats[field]["nonexact"],
                "meaningful_mismatches": stats[field]["meaningful_mismatch"],
                "max_absolute_difference": maximum[field],
                "tolerance": 5e-5 if field in CLIMATE else 0,
            })
    difference_path = args.output_dir / "commune_panel_vs_mongo_2026_differences.csv"
    with difference_path.open("w", newline="", encoding="utf-8") as handle:
        columns = [
            "fcode", "year", "month", "field", "panel_value", "mongo_value",
            "absolute_difference",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(differences)
    print(json.dumps({
        "overlap_rows": len(set(panel) & set(mongo)),
        "overlap_months": sorted({f"{year}-{month:02d}" for _, year, month in set(panel) & set(mongo)}),
        "meaningful_differences": len(differences),
        "differences_by_field": dict(Counter(row["field"] for row in differences)),
    }, indent=2))


if __name__ == "__main__":
    main()
