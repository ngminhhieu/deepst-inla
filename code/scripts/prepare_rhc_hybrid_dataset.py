#!/usr/bin/env python3
"""Prepare the 128-RHC panel, graph, MOH thresholds and province labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXTRA = ["soil_moisture", "solar_downward", "dewpoint_c", "et0_mm"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rhc-panel", type=Path, required=True)
    parser.add_argument("--commune-panel", type=Path, required=True)
    parser.add_argument("--mongo-export", type=Path, required=True)
    parser.add_argument("--commune-centroids", type=Path, required=True)
    parser.add_argument("--commune-adjacency", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mongo = json.loads(args.mongo_export.read_text(encoding="utf-8"))
    mapping_rows = []
    for row in mongo["boundary"]:
        props = row.get("properties") or {}
        mapping_rows.append({
            "commune_id": str(row.get("fcode")), "fcode": props.get("ttytkv"),
            "l1_code": str(props.get("l1_code")), "province": props.get("l1_name_loc"),
        })
    mapping = pd.DataFrame(mapping_rows).dropna(subset=["fcode"]).drop_duplicates("commune_id")
    if mapping.groupby("fcode").province.nunique().max() != 1:
        raise ValueError("An RHC maps to multiple provinces")
    rhc_labels = mapping.groupby("fcode", as_index=False).agg(
        province=("province", "first"), province_code=("l1_code", "first")
    )

    rhc = pd.read_csv(args.rhc_panel, dtype={"fcode": str, "l1_code": str})
    commune = pd.read_csv(args.commune_panel, dtype={"fcode": str})
    commune = commune.merge(
        mapping[["commune_id", "fcode"]], left_on="fcode", right_on="commune_id",
        how="inner", suffixes=("_commune", "_rhc"), validate="many_to_one",
    )
    commune["weight"] = pd.to_numeric(commune["population"], errors="coerce").clip(lower=1)
    weighted_rows = []
    for (rhc_id, year, month), part in commune.groupby(["fcode_rhc", "year", "month"], sort=False):
        weights = part["weight"].to_numpy(float)
        row = {"fcode": rhc_id, "year": int(year), "month": int(month)}
        for column in EXTRA:
            row[column] = float(np.average(part[column].to_numpy(float), weights=weights))
        weighted_rows.append(row)
    external = pd.DataFrame(weighted_rows)
    rhc = rhc.merge(external, on=["fcode", "year", "month"], how="left", validate="one_to_one")
    rhc = rhc.merge(rhc_labels, on="fcode", how="left", validate="many_to_one")
    if rhc[EXTRA + ["province"]].isna().any().any():
        raise ValueError("RHC panel has missing external features or province labels")
    rhc = rhc.sort_values(["fcode", "year", "month"])
    rhc.to_csv(args.output_dir / "monthly_rhc_through_2026-08.csv", index=False)
    rhc_labels.to_csv(args.output_dir / "rhc_province_lookup.csv", index=False)

    threshold_rows = []
    for fcode, frame in rhc.groupby("fcode", sort=True):
        values = {(int(row.year), int(row.month)): float(row.obs_dengue_cases) for row in frame.itertuples()}
        for row in frame.itertuples():
            history = [values.get((int(row.year) - lag, int(row.month))) for lag in range(1, 6)]
            if any(value is None for value in history):
                threshold = np.nan
            else:
                threshold = float(np.mean(history) + 2 * np.std(history, ddof=1))
            threshold_rows.append({
                "fcode": fcode, "year": int(row.year), "month": int(row.month),
                "alert_threshold_moh": threshold,
            })
    thresholds = pd.DataFrame(threshold_rows).dropna(subset=["alert_threshold_moh"])
    thresholds.to_csv(args.output_dir / "moh_thresholds_rhc_2015_2026.csv", index=False)

    centroids = pd.read_csv(args.commune_centroids, dtype={"fcode": str}).merge(
        mapping[["commune_id", "fcode"]], left_on="fcode", right_on="commune_id",
        how="inner", suffixes=("_commune", "_rhc"), validate="one_to_one",
    )
    latest_population = (
        commune.sort_values(["year", "month"]).groupby("fcode_commune", as_index=False).tail(1)
        [["fcode_commune", "population"]]
    )
    centroids = centroids.merge(latest_population, on="fcode_commune", how="left")
    centroid_rows = []
    for fcode, part in centroids.groupby("fcode_rhc", sort=True):
        weights = part.population.to_numpy(float).clip(min=1)
        centroid_rows.append({
            "fcode": fcode,
            "longitude": float(np.average(part.longitude, weights=weights)),
            "latitude": float(np.average(part.latitude, weights=weights)),
        })
    pd.DataFrame(centroid_rows).to_csv(args.output_dir / "rhc_centroids.csv", index=False)

    commune_to_rhc = mapping.set_index("commune_id")["fcode"].to_dict()
    adjacency = pd.read_csv(args.commune_adjacency, dtype=str)
    edges = set()
    for row in adjacency.itertuples(index=False):
        left, right = commune_to_rhc.get(str(row[0])), commune_to_rhc.get(str(row[1]))
        if left and right and left != right:
            edges.add((left, right))
    pd.DataFrame(sorted(edges), columns=["from", "to"]).to_csv(
        args.output_dir / "rhc_adjacency.csv", index=False
    )
    manifest = {
        "rhc_units": int(rhc.fcode.nunique()), "rows": len(rhc),
        "date_start": f"{int(rhc.year.min()):04d}-01",
        "date_end": "2026-08", "provinces": int(rhc.province.nunique()),
        "adjacency_directed_edges": len(edges),
        "threshold_definition": "same calendar month in previous 5 years: mean + 2 sample SD",
        "external_features": "population-weighted aggregation of commune public covariates",
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
