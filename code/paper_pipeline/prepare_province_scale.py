#!/usr/bin/env python3
"""Build the six-province panel and graph used by the cross-scale experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLUMNS = ["fcode", "year", "month", "obs_dengue_cases", "population", "l1_code"]
WEIGHTED_COLUMNS = [
    "ws_avg", "ws_min", "ws_max", "t2m_avg", "t2m_min", "t2m_max",
    "rh_avg", "rh_min", "rh_max", "tp_accum",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_panel(source: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(ID_COLUMNS + WEIGHTED_COLUMNS).difference(source.columns))
    if missing:
        raise ValueError(f"Province source is missing columns: {missing}")
    data = source.copy()
    data["date"] = pd.to_datetime(dict(year=data.year, month=data.month, day=1))
    if data.duplicated(["fcode", "date"]).any():
        raise ValueError("Source contains duplicate spatial-unit months")
    for column in ["obs_dengue_cases", "population", *WEIGHTED_COLUMNS]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[["obs_dengue_cases", "population", *WEIGHTED_COLUMNS]].isna().any().any():
        raise ValueError("Source contains missing values in required numeric fields")
    if (data["obs_dengue_cases"] < 0).any() or (data["population"] <= 0).any():
        raise ValueError("Cases must be non-negative and population positive")
    mapping_counts = data.groupby("fcode")["l1_code"].nunique()
    if (mapping_counts != 1).any():
        raise ValueError("A source unit maps to multiple provinces")

    rows = []
    for (province, year, month), part in data.groupby(["l1_code", "year", "month"], sort=True):
        weights = part["population"].to_numpy(float)
        row = {
            "fcode": f"PROV_{province}",
            "l1_code": "STUDY_REGION",
            "province_code": str(province),
            "year": int(year),
            "month": int(month),
            "obs_dengue_cases": float(part["obs_dengue_cases"].sum()),
            "population": float(part["population"].sum()),
            "area": float(part["area"].sum()) if "area" in part else np.nan,
        }
        for column in WEIGHTED_COLUMNS:
            row[column] = float(np.average(part[column], weights=weights))
        rows.append(row)
    output = pd.DataFrame(rows).sort_values(["year", "month", "fcode"]).reset_index(drop=True)
    observed_dates = pd.to_datetime(dict(year=output.year, month=output.month, day=1))
    dates = pd.date_range(observed_dates.min(), observed_dates.max(), freq="MS")
    expected = len(dates) * output.fcode.nunique()
    if len(output) != expected:
        raise ValueError(f"Aggregated province panel is incomplete: {len(output)} != {expected}")
    return output


def build_spatial_assets(
    commune_monthly: pd.DataFrame,
    commune_centroids: pd.DataFrame,
    commune_adjacency: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = commune_monthly[["fcode", "l1_code"]].drop_duplicates().copy()
    mapping["fcode"] = mapping["fcode"].astype(str)
    mapping["province_fcode"] = "PROV_" + mapping["l1_code"].astype(str)
    if mapping.groupby("fcode").province_fcode.nunique().max() != 1:
        raise ValueError("Commune reference has non-unique province assignments")

    latest = commune_monthly.copy()
    latest["date"] = pd.to_datetime(dict(year=latest.year, month=latest.month, day=1))
    latest = latest.sort_values("date").groupby("fcode", as_index=False).tail(1)
    latest["fcode"] = latest["fcode"].astype(str)
    points = commune_centroids.copy()
    points["fcode"] = points["fcode"].astype(str)
    points = points.merge(mapping[["fcode", "province_fcode"]], on="fcode", validate="one_to_one")
    points = points.merge(latest[["fcode", "population"]], on="fcode", validate="one_to_one")
    centroids = []
    for province, part in points.groupby("province_fcode", sort=True):
        weights = part.population.to_numpy(float)
        centroids.append({
            "fcode": province,
            "longitude": float(np.average(part.longitude, weights=weights)),
            "latitude": float(np.average(part.latitude, weights=weights)),
        })
    centroids = pd.DataFrame(centroids)

    aliases = [("from", "to"), ("from_fcode", "to_fcode"), ("source", "target")]
    edge_columns = next((pair for pair in aliases if set(pair) <= set(commune_adjacency)), None)
    if edge_columns is None:
        raise ValueError("Unrecognised commune adjacency columns")
    lookup = mapping.set_index("fcode").province_fcode
    edges = commune_adjacency[list(edge_columns)].astype(str).rename(
        columns={edge_columns[0]: "from", edge_columns[1]: "to"}
    )
    edges["from"] = edges["from"].map(lookup)
    edges["to"] = edges["to"].map(lookup)
    edges = edges.dropna()
    edges = edges[edges["from"] != edges["to"]]
    edges = edges.drop_duplicates().sort_values(["from", "to"]).reset_index(drop=True)
    return centroids, edges


def source_comparison(province: pd.DataFrame, commune: pd.DataFrame) -> dict[str, float | int | str]:
    left = province.copy()
    left["province_code"] = left["province_code"].astype(str)
    right = commune.groupby(["l1_code", "year", "month"], as_index=False).agg(
        commune_cases=("obs_dengue_cases", "sum"), commune_population=("population", "sum")
    )
    right["province_code"] = right.l1_code.astype(str)
    merged = left.merge(right, on=["province_code", "year", "month"], how="inner")
    case_difference = merged.obs_dengue_cases - merged.commune_cases
    return {
        "overlap_rows": int(len(merged)),
        "case_exact_rows": int(np.isclose(case_difference, 0).sum()),
        "case_correlation": float(merged.obs_dengue_cases.corr(merged.commune_cases)),
        "case_mean_absolute_difference": float(case_difference.abs().mean()),
        "case_max_absolute_difference": float(case_difference.abs().max()),
        "population_correlation": float(merged.population.corr(merged.commune_population)),
        "interpretation": "same_underlying_surveillance_not_independent",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--commune-monthly", type=Path, required=True)
    parser.add_argument("--commune-centroids", type=Path, required=True)
    parser.add_argument("--commune-adjacency", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = pd.read_csv(args.source, dtype={"fcode": str, "l1_code": str})
    commune = pd.read_csv(args.commune_monthly, dtype={"fcode": str, "l1_code": str})
    province = aggregate_panel(source)
    centroids, adjacency = build_spatial_assets(
        commune,
        pd.read_csv(args.commune_centroids, dtype={"fcode": str}),
        pd.read_csv(args.commune_adjacency, dtype=str),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    province.to_csv(args.output_dir / "monthly_province.csv", index=False)
    centroids.to_csv(args.output_dir / "province_centroids.csv", index=False)
    adjacency.to_csv(args.output_dir / "province_adjacency.csv", index=False)
    dates = pd.to_datetime(dict(year=province.year, month=province.month, day=1))
    audit = {
        "source_file_name": args.source.name,
        "source_sha256": sha256(args.source),
        "source_rows": int(len(source)),
        "source_spatial_units": int(source.fcode.nunique()),
        "province_units": int(province.fcode.nunique()),
        "date_start": str(dates.min().date()),
        "date_end": str(dates.max().date()),
        "province_rows": int(len(province)),
        "missing_required_values": 0,
        "duplicate_unit_months": 0,
        "environmental_variables": WEIGHTED_COLUMNS,
        "aggregation": "cases/population/area summed; environmental fields population-weighted",
        "province_adjacency_directed_edges": int(len(adjacency)),
        "source_independence_check": source_comparison(province, commune),
    }
    (args.output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
