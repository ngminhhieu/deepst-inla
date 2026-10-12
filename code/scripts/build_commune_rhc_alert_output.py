#!/usr/bin/env python3
"""Convert commune forecasts to MOH alerts and compare with stored RHC alerts.

MongoDB is read only.  RHC alerts are never reconstructed: the script uses the
stored ``intervention_flag`` for the matching target month and model horizon,
and leaves unavailable values as NA.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pymongo import MongoClient


OUTPUT_COLUMNS = [
    "forecast_month",
    "commune_id",
    "commune_name",
    "rhc_id",
    "rhc_name",
    "commune_prediction",
    "commune_alert",
    "rhc_alert",
    "commune_rhc_match",
    "comparison_category",
]


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return f"external:{resolved.name}"


def clean_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def rhc_display_name(rhc_id: str) -> str:
    label = re.sub(r"^TTYTKV_", "", rhc_id)
    parts = label.split("_")
    if len(parts) > 1 and len(parts[-1]) <= 3:
        parts = parts[:-1]
    return "TTYT khu vực " + " ".join(parts).title()


def load_predictions(predictions_path: Path, samples_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(predictions_path, dtype={"fcode": str})
    required = {"fcode", "target_date", "horizon"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction file is missing columns: {missing}")
    frame["fcode"] = frame["fcode"].map(clean_id)
    frame["target_date"] = pd.to_datetime(frame["target_date"]).dt.to_period("M").dt.to_timestamp()
    frame["horizon"] = frame["horizon"].astype(int)
    cube = np.load(samples_path)["samples"]
    if cube.ndim != 4:
        raise ValueError(f"Expected a 4-D predictive sample cube, found {cube.shape}")
    return frame, cube


def flatten_forecast(frame: pd.DataFrame, cube: np.ndarray) -> pd.DataFrame:
    rows = []
    horizons = sorted(frame["horizon"].unique())
    if cube.shape[2] != len(horizons):
        raise ValueError("Prediction horizons do not match the predictive sample cube")
    for index, horizon in enumerate(horizons):
        part = frame.loc[frame["horizon"].eq(horizon)].reset_index(drop=True).copy()
        values = cube[:, :, index, :].reshape(-1, cube.shape[-1])
        if len(part) != len(values):
            raise ValueError(f"H{horizon} prediction/sample row mismatch")
        part["commune_prediction"] = values.mean(axis=1)
        part["_samples"] = list(values)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def load_thresholds(path: Path) -> pd.DataFrame:
    if "ytkv" in path.name.lower():
        raise ValueError("RHC/YTKV threshold files are forbidden for commune alerts")
    frame = pd.read_csv(path, dtype={"fcode": str})
    required = {"fcode", "year", "month", "alert_threshold_moh"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Commune threshold file is missing columns: {missing}")
    frame["fcode"] = frame["fcode"].map(clean_id)
    frame["target_date"] = pd.to_datetime(
        dict(year=frame["year"], month=frame["month"], day=1)
    )
    if frame.duplicated(["fcode", "target_date"]).any():
        raise ValueError("Duplicate commune-month thresholds")
    frame["alert_threshold_moh"] = pd.to_numeric(
        frame["alert_threshold_moh"], errors="coerce"
    )
    return frame[["fcode", "target_date", "alert_threshold_moh"]]


def mongo_reference(
    uri: str,
    database: str,
    boundary_collection: str,
    rhc_collection: str,
    year: int,
) -> tuple[pd.DataFrame, dict[tuple[str, int, int], int]]:
    client = MongoClient(uri, serverSelectionTimeoutMS=30_000)
    db = client[database]
    mapping_rows = []
    for doc in db[boundary_collection].find(
        {}, {"fcode": 1, "properties.l2_name_loc": 1, "properties.ttytkv": 1}
    ):
        props = doc.get("properties") or {}
        mapping_rows.append(
            {
                "fcode": clean_id(doc.get("fcode")),
                "commune_name": props.get("l2_name_loc"),
                "rhc_id": props.get("ttytkv"),
            }
        )
    mapping = pd.DataFrame(mapping_rows)
    if mapping.empty or mapping.duplicated("fcode").any():
        raise ValueError("MongoDB commune-to-RHC mapping is empty or duplicated")
    mapping["rhc_name"] = mapping["rhc_id"].fillna("").map(rhc_display_name)

    alerts: dict[tuple[str, int, int], int] = {}
    for doc in db[rhc_collection].find({"year": int(year)}, {"fcode": 1, "months": 1}):
        rhc_id = str(doc.get("fcode", ""))
        for month_text, month_doc in (doc.get("months") or {}).items():
            for horizon_text, values in (month_doc.get("horizons") or {}).items():
                flag = values.get("intervention_flag")
                if flag in (0, 1):
                    alerts[(rhc_id, int(month_text), int(horizon_text))] = int(flag)
    client.close()
    return mapping, alerts


def category(commune: int, rhc: object) -> str:
    if pd.isna(rhc):
        return "rhc_alert_unavailable"
    if commune == 1 and int(rhc) == 1:
        return "both_alerted"
    if commune == 1:
        return "commune_only"
    if int(rhc) == 1:
        return "rhc_only"
    return "neither_alerted"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--probability-cutoff", type=float, default=0.50)
    parser.add_argument("--mongo-database", default="edengue")
    parser.add_argument("--boundary-collection", default="ed_boundary_mdr_lev2")
    parser.add_argument("--rhc-collection", default="ed_forecast_mdr_ytkv")
    args = parser.parse_args()

    uri = os.environ.get("DENGUE_MONGODB_URI", "").strip()
    if not uri:
        raise RuntimeError("DENGUE_MONGODB_URI is required")
    frame, cube = load_predictions(args.predictions, args.samples)
    forecast = flatten_forecast(frame, cube)
    thresholds = load_thresholds(args.thresholds)
    forecast_dates = set(forecast["target_date"].unique())
    threshold_scope = thresholds.loc[thresholds["target_date"].isin(forecast_dates)]
    threshold_communes = set(threshold_scope["fcode"].unique())
    model_communes = set(forecast["fcode"].unique())
    communes_without_model_prediction = sorted(threshold_communes - model_communes)
    forecast = forecast.merge(
        thresholds, on=["fcode", "target_date"], how="left", validate="many_to_one"
    )
    if forecast["alert_threshold_moh"].isna().any():
        missing = forecast.loc[
            forecast["alert_threshold_moh"].isna(), ["fcode", "target_date"]
        ].head()
        raise ValueError(f"Missing commune MOH thresholds:\n{missing}")
    forecast["commune_alert_probability"] = [
        float(np.mean(np.asarray(samples) >= threshold))
        for samples, threshold in zip(forecast["_samples"], forecast["alert_threshold_moh"])
    ]
    forecast["commune_alert"] = (
        forecast["commune_alert_probability"] >= args.probability_cutoff
    ).astype("Int64")

    forecast_years = sorted(forecast["target_date"].dt.year.unique())
    if len(forecast_years) != 1:
        raise ValueError(f"Expected one forecast year, found {forecast_years}")
    mapping, rhc_alerts = mongo_reference(
        uri,
        args.mongo_database,
        args.boundary_collection,
        args.rhc_collection,
        int(forecast_years[0]),
    )
    forecast = forecast.merge(mapping, on="fcode", how="left", validate="many_to_one")
    if forecast["rhc_id"].isna().any():
        missing = forecast.loc[forecast["rhc_id"].isna(), "fcode"].unique()[:10]
        raise ValueError(f"MongoDB has no parent RHC mapping for communes: {missing}")
    forecast["rhc_alert"] = pd.array(
        [
            rhc_alerts.get((str(row.rhc_id), int(row.target_date.month), int(row.horizon)))
            for row in forecast.itertuples(index=False)
        ],
        dtype="Int64",
    )
    forecast["commune_rhc_match"] = pd.array(
        [
            pd.NA if pd.isna(rhc) else int(int(commune) == 1 and int(rhc) == 1)
            for commune, rhc in zip(forecast["commune_alert"], forecast["rhc_alert"])
        ],
        dtype="Int64",
    )
    forecast["comparison_category"] = [
        category(int(commune), rhc)
        for commune, rhc in zip(forecast["commune_alert"], forecast["rhc_alert"])
    ]
    forecast["forecast_month"] = forecast["target_date"].dt.strftime("%Y-%m")
    forecast = forecast.rename(columns={"fcode": "commune_id"})
    output = forecast[OUTPUT_COLUMNS].sort_values(["forecast_month", "commune_id"])
    if output.duplicated(["forecast_month", "commune_id"]).any():
        raise ValueError("Output is not unique by forecast_month x commune")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, na_rep="NA")

    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = {
        "model_predictions": portable_path(args.predictions),
        "model_samples": portable_path(args.samples),
        "threshold_file": portable_path(args.thresholds),
        "threshold_column": "alert_threshold_moh",
        "forbidden_threshold_source_used": False,
        "commune_alert_rule": (
            "P(predictive cases >= alert_threshold_moh) >= "
            f"{args.probability_cutoff:.2f}"
        ),
        "mongo_database": args.mongo_database,
        "boundary_collection": args.boundary_collection,
        "rhc_alert_collection": args.rhc_collection,
        "rhc_alert_field": "months.<month>.horizons.<horizon>.intervention_flag",
        "rhc_alert_reconstructed": False,
        "rows": len(output),
        "communes": int(output["commune_id"].nunique()),
        "threshold_communes_in_forecast_months": len(threshold_communes),
        "model_supported_communes": len(model_communes),
        "communes_without_model_prediction": communes_without_model_prediction,
        "forecast_months": sorted(output["forecast_month"].unique().tolist()),
        "rhc_alert_available_rows": int(output["rhc_alert"].notna().sum()),
        "rhc_alert_unavailable_rows": int(output["rhc_alert"].isna().sum()),
        "comparison_category_counts": {
            str(key): int(value)
            for key, value in output["comparison_category"].value_counts(dropna=False).items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
