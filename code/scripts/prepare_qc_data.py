#!/usr/bin/env python3
"""Audit commune data and materialise a reproducible analysis-ready panel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = ROOT.parent / "dengue_forecast_v1/dengue_training/data"
CLIMATE_COLUMNS = ("t2m_avg", "rh_avg", "tp_accum")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_communes(
    monthly: pd.DataFrame,
    names: pd.DataFrame,
    *,
    minimum_population: float,
    minimum_density: float,
    maximum_monthly_incidence: float,
) -> pd.DataFrame:
    frame = monthly.copy()
    frame["date"] = pd.to_datetime(
        {"year": frame["year"], "month": frame["month"], "day": 1}
    )
    expected_months = len(pd.date_range(frame.date.min(), frame.date.max(), freq="MS"))
    frame["monthly_incidence"] = (
        frame["obs_dengue_cases"] / frame["population"] * 100_000
    )
    duplicate_units = set(
        frame.loc[frame.duplicated(["fcode", "date"], keep=False), "fcode"]
    )
    missing_feature_units = set(
        frame.loc[
            frame[["obs_dengue_cases", "population", *CLIMATE_COLUMNS]].isna().any(axis=1),
            "fcode",
        ]
    )
    invalid_case_units = set(frame.loc[frame["obs_dengue_cases"] < 0, "fcode"])
    invalid_population_units = set(frame.loc[frame["population"] <= 0, "fcode"])

    summary = (
        frame.groupby("fcode", as_index=False)
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            observed_months=("date", "nunique"),
            row_count=("date", "size"),
            population_min=("population", "min"),
            population_median=("population", "median"),
            population_max=("population", "max"),
            area_median=("area", "median"),
            cases_max=("obs_dengue_cases", "max"),
            cases_total=("obs_dengue_cases", "sum"),
            monthly_incidence_max=("monthly_incidence", "max"),
        )
        .sort_values("fcode")
        .reset_index(drop=True)
    )
    summary["population_density_median"] = (
        summary["population_median"] / summary["area_median"]
    )
    summary["population_fold_change"] = (
        summary["population_max"] / summary["population_min"]
    )
    summary = summary.merge(
        names[["fcode", "region_name", "province_name"]].drop_duplicates("fcode"),
        on="fcode",
        how="left",
    )

    reasons: list[str] = []
    for row in summary.itertuples(index=False):
        unit_reasons = []
        if row.fcode in duplicate_units:
            unit_reasons.append("duplicate_commune_month")
        if row.observed_months != expected_months or row.row_count != expected_months:
            unit_reasons.append("incomplete_monthly_panel")
        if row.fcode in missing_feature_units:
            unit_reasons.append("missing_required_feature")
        if row.fcode in invalid_case_units:
            unit_reasons.append("negative_case_count")
        if row.fcode in invalid_population_units:
            unit_reasons.append("nonpositive_population")
        if row.population_min < minimum_population:
            unit_reasons.append("population_below_minimum")
        if row.population_density_median < minimum_density:
            unit_reasons.append("population_density_below_minimum")
        if row.monthly_incidence_max > maximum_monthly_incidence:
            unit_reasons.append("monthly_incidence_above_maximum")
        reasons.append(";".join(unit_reasons))
    summary["qc_reasons"] = reasons
    summary["qc_exclude"] = summary["qc_reasons"].ne("")
    summary["expected_months"] = expected_months
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--monthly",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "processed/monthly_commune_from_mongodb.csv",
    )
    parser.add_argument(
        "--centroids",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "static/inla/commune_centroids.csv",
    )
    parser.add_argument(
        "--adjacency",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "static/inla/administrative_adjacency.csv",
    )
    parser.add_argument(
        "--names",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "static/commune_names.csv",
    )
    parser.add_argument(
        "--data-output", type=Path, default=ROOT / "code/data/processed/qc_v1"
    )
    parser.add_argument(
        "--report-output", type=Path, default=ROOT / "results/qc_v1/data_quality"
    )
    parser.add_argument("--minimum-population", type=float, default=1_000)
    parser.add_argument("--minimum-density", type=float, default=5)
    parser.add_argument("--maximum-monthly-incidence", type=float, default=5_000)
    args = parser.parse_args()

    monthly = pd.read_csv(args.monthly, dtype={"fcode": str})
    centroids = pd.read_csv(args.centroids, dtype={"fcode": str})
    adjacency = pd.read_csv(args.adjacency, dtype=str)
    names = pd.read_csv(args.names, dtype={"fcode": str})
    report = audit_communes(
        monthly,
        names,
        minimum_population=args.minimum_population,
        minimum_density=args.minimum_density,
        maximum_monthly_incidence=args.maximum_monthly_incidence,
    )
    excluded = set(report.loc[report.qc_exclude, "fcode"])
    retained = set(report.loc[~report.qc_exclude, "fcode"])
    if not excluded:
        raise SystemExit("QC excluded no communes; inspect thresholds before proceeding")

    edge_aliases = (("from", "to"), ("from_fcode", "to_fcode"), ("source", "target"))
    edge_columns = next(
        ((left, right) for left, right in edge_aliases if {left, right} <= set(adjacency)),
        None,
    )
    if edge_columns is None:
        raise ValueError("Unrecognised adjacency columns")
    filtered_monthly = monthly[monthly.fcode.isin(retained)].copy()
    filtered_centroids = centroids[centroids.fcode.isin(retained)].copy()
    filtered_adjacency = adjacency[
        adjacency[edge_columns[0]].isin(retained)
        & adjacency[edge_columns[1]].isin(retained)
    ].copy()

    args.data_output.mkdir(parents=True, exist_ok=True)
    args.report_output.mkdir(parents=True, exist_ok=True)
    monthly_output = args.data_output / "monthly_commune.csv"
    centroids_output = args.data_output / "commune_centroids.csv"
    adjacency_output = args.data_output / "administrative_adjacency.csv"
    filtered_monthly.to_csv(monthly_output, index=False)
    filtered_centroids.to_csv(centroids_output, index=False)
    filtered_adjacency.to_csv(adjacency_output, index=False)
    report.to_csv(args.report_output / "commune_qc.csv", index=False)
    report[report.qc_exclude].to_csv(
        args.report_output / "excluded_communes.csv", index=False
    )

    manifest = {
        "qc_version": "qc_v1",
        "criteria": {
            "minimum_population_any_month": args.minimum_population,
            "minimum_median_population_density_per_km2": args.minimum_density,
            "maximum_monthly_incidence_per_100000": args.maximum_monthly_incidence,
            "require_complete_panel": True,
            "reject_duplicates_missing_required_features_negative_cases": True,
        },
        "source": {
            "monthly": f"external_input/{args.monthly.name}",
            "monthly_sha256": sha256(args.monthly),
            "centroids": f"external_input/{args.centroids.name}",
            "adjacency": f"external_input/{args.adjacency.name}",
            "names": f"external_input/{args.names.name}",
        },
        "communes_input": int(monthly.fcode.nunique()),
        "communes_retained": len(retained),
        "communes_excluded": len(excluded),
        "excluded_fcodes": sorted(excluded),
        "rows_output": len(filtered_monthly),
        "output_sha256": sha256(monthly_output),
    }
    (args.report_output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"retained {len(retained)}/{monthly.fcode.nunique()} communes; "
        f"excluded {', '.join(sorted(excluded))}"
    )
    print(report.loc[report.qc_exclude, [
        "fcode", "region_name", "population_min", "population_median",
        "population_density_median", "monthly_incidence_max", "qc_reasons",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
