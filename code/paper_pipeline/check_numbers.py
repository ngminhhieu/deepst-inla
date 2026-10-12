#!/usr/bin/env python3
"""Check mapped manuscript numbers against their authoritative CSV cells."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
NUMBER = re.compile(r"(?<![A-Za-z])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")


def equal_filter(series, expected):
    numeric = pd.to_numeric(series, errors="coerce")
    try:
        value = float(expected)
    except (TypeError, ValueError):
        return series.astype(str) == str(expected)
    if numeric.notna().all():
        return (numeric - value).abs() < 1e-12
    return series.astype(str) == str(expected)


def significant_digits(token):
    digits = re.sub(r"\D", "", token.split("e")[0].split("E")[0])
    return len(digits.lstrip("0"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, default=PAPER / "number_sources.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.mapping.read_text())
    failures, mapped = [], {}
    categories = set()
    for check in config.get("checks", []):
        categories.add(check["category"])
        tex_path = PAPER / check["file"]
        content = tex_path.read_text()
        match = re.search(check["pattern"], content, flags=re.DOTALL)
        if not match:
            failures.append(f"{check['id']}: pattern not found in {tex_path.relative_to(ROOT)}")
            continue
        observed_text = match.group("value")
        observed = float(observed_text.replace(",", ""))
        csv_path = (PAPER / check["csv"]).resolve()
        frame = pd.read_csv(csv_path)
        for column, value in check.get("filters", {}).items():
            frame = frame[equal_filter(frame[column], value)]
        if frame.empty:
            failures.append(f"{check['id']}: CSV filter returned no rows")
            continue
        values = pd.to_numeric(frame[check["column"]], errors="raise")
        aggregate = check.get("aggregate")
        if aggregate == "mean":
            expected = float(values.mean())
        elif len(values) == 1:
            expected = float(values.iloc[0])
        else:
            failures.append(f"{check['id']}: CSV filter returned {len(values)} rows without aggregate")
            continue
        expected *= float(check.get("scale", 1))
        digits = int(check["rounding"])
        if round(observed, digits) != round(expected, digits):
            failures.append(f"{check['id']}: manuscript {observed} != CSV {expected:.12g} at {digits} decimals")
        mapped.setdefault(tex_path.resolve(), []).append(match.span("value"))

    required = {"abstract", "table1", "horizon_sensitivities", "event_counts",
                "paired_contrasts", "spatial_support", "threshold_sensitivity",
                "decision_utility", "timesfm", "holdout", "seed_figures",
                "external_transfer"}
    missing = sorted(required - categories)
    if missing:
        failures.append("mapping categories missing: " + ", ".join(missing))

    warnings = []
    tex_files = sorted((PAPER / "sections").glob("*.tex")) + sorted(
        (PAPER / "supplement_sections").glob("*.tex")) + sorted((PAPER / "generated").glob("*.tex"))
    for path in tex_files:
        content = path.read_text()
        spans = mapped.get(path.resolve(), [])
        for match in NUMBER.finditer(content):
            if significant_digits(match.group()) <= 2 or any(a <= match.start() and match.end() <= b for a, b in spans):
                continue
            line = content.count("\n", 0, match.start()) + 1
            warnings.append(f"WARN unmapped {path.relative_to(ROOT)}:{line}: {match.group()}")
    print(f"checked {len(config.get('checks', []))} mappings across {len(categories)} required categories")
    for warning in warnings:
        print(warning)
    print(f"unmapped numeric warnings (>2 significant digits): {len(warnings)}")
    if failures:
        for failure in failures:
            print("ERROR", failure, file=sys.stderr)
        return 1
    print("number consistency: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
