#!/usr/bin/env python3
"""Build leakage-safe derived commune-month features from data already in the project.

The output keeps every original column and adds only features that can be computed
at or before each issue month. It does *not* use future realised climate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def haversine_matrix(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat = np.radians(latitude.astype(float))
    lon = np.radians(longitude.astype(float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def add_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["fcode", "date"]).copy()
    if "area" in frame.columns:
        area = pd.to_numeric(frame["area"], errors="coerce")
        if area.notna().all() and (area > 0).all():
            frame["population_density"] = frame["population"] / area
            frame["log_population_density"] = np.log1p(frame["population_density"])

    if "tp_accum" in frame.columns:
        group = frame.groupby("fcode", sort=False)["tp_accum"]
        frame["rain_3m_sum"] = group.transform(lambda s: s.rolling(3, min_periods=1).sum())
        frame["rain_6m_sum"] = group.transform(lambda s: s.rolling(6, min_periods=1).sum())
    if "t2m_avg" in frame.columns:
        group = frame.groupby("fcode", sort=False)["t2m_avg"]
        frame["temp_3m_mean"] = group.transform(lambda s: s.rolling(3, min_periods=1).mean())
    if "rh_avg" in frame.columns:
        group = frame.groupby("fcode", sort=False)["rh_avg"]
        frame["humidity_3m_mean"] = group.transform(lambda s: s.rolling(3, min_periods=1).mean())

    cases = frame.groupby("fcode", sort=False)["obs_dengue_cases"]
    lag1 = cases.shift(1)
    lag3 = cases.shift(3)
    frame["case_growth_1m"] = np.log1p(frame["obs_dengue_cases"]) - np.log1p(lag1.fillna(frame["obs_dengue_cases"]))
    frame["case_growth_3m"] = np.log1p(frame["obs_dengue_cases"]) - np.log1p(lag3.fillna(frame["obs_dengue_cases"]))
    return frame


def add_neighbor_features(frame: pd.DataFrame, adjacency: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    incidence = np.log1p(output["obs_dengue_cases"] / output["population"] * 100_000.0)
    output["_log_incidence"] = incidence
    pivoted = output.pivot(index="date", columns="fcode", values="_log_incidence")
    units = list(pivoted.columns)
    neighbor_map: dict[str, list[str]] = {unit: [] for unit in units}
    edges = adjacency.copy()
    edges["from"] = edges["from"].astype(str)
    edges["to"] = edges["to"].astype(str)
    for source, target in zip(edges["from"], edges["to"]):
        if source in neighbor_map and target in neighbor_map and source != target:
            neighbor_map[source].append(target)
    mean_values = pd.DataFrame(index=pivoted.index, columns=units, dtype=float)
    max_values = pd.DataFrame(index=pivoted.index, columns=units, dtype=float)
    for unit, neighbors in neighbor_map.items():
        if not neighbors:
            mean_values[unit] = 0.0
            max_values[unit] = 0.0
            continue
        block = pivoted.reindex(columns=neighbors)
        mean_values[unit] = block.mean(axis=1).fillna(0.0)
        max_values[unit] = block.max(axis=1).fillna(0.0)
    mean_long = mean_values.stack().rename("neighbor_mean_log_incidence").reset_index()
    max_long = max_values.stack().rename("neighbor_max_log_incidence").reset_index()
    mean_long.columns = ["date", "fcode", "neighbor_mean_log_incidence"]
    max_long.columns = ["date", "fcode", "neighbor_max_log_incidence"]
    output = output.merge(mean_long, on=["date", "fcode"], how="left")
    output = output.merge(max_long, on=["date", "fcode"], how="left")
    return output.drop(columns="_log_incidence")


def add_climate_anomalies(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    mapping = {
        "t2m_avg": "t2m_anomaly",
        "tp_accum": "tp_anomaly",
        "rh_avg": "rh_anomaly",
    }
    for source, destination in mapping.items():
        if source not in output.columns:
            continue
        climatology = (
            output.groupby(["fcode", "month"])[source]
            .transform(lambda series: series.shift(1).expanding(min_periods=3).mean())
        )
        output[destination] = (output[source] - climatology).fillna(0.0)
    return output


def add_province_loo_incidence(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    incidence = np.log1p(output["obs_dengue_cases"] / output["population"] * 100_000.0)
    grouped = output.assign(_inc=incidence).groupby(["date", "l1_code"], sort=False)
    total = grouped["_inc"].transform("sum")
    count = grouped["_inc"].transform("size")
    output["province_loo_log_incidence"] = np.where(
        count > 1, (total - incidence) / (count - 1), incidence
    )
    return output


def add_gravity_import_pressure(
    frame: pd.DataFrame,
    centroids: pd.DataFrame,
    *,
    distance_power: float = 2.0,
    minimum_distance_km: float = 5.0,
    top_k: int = 50,
) -> pd.DataFrame:
    units = tuple(sorted(frame["fcode"].astype(str).unique()))
    centroids = centroids.copy()
    centroids["fcode"] = centroids["fcode"].astype(str)
    centroids = centroids.set_index("fcode").reindex(units)
    if centroids[["latitude", "longitude"]].isna().any().any():
        missing = centroids.index[centroids["latitude"].isna() | centroids["longitude"].isna()].tolist()[:10]
        raise ValueError(f"Missing centroids for communes: {missing}")
    distance = haversine_matrix(centroids["latitude"].to_numpy(), centroids["longitude"].to_numpy())
    distance = np.maximum(distance, minimum_distance_km)
    np.fill_diagonal(distance, np.inf)

    output = frame.copy()
    pressure_by_date = []
    for date, part in output.groupby("date", sort=True):
        indexed = part.set_index("fcode").reindex(units)
        population = indexed["population"].to_numpy(float)
        incidence = indexed["obs_dengue_cases"].to_numpy(float) / population * 100_000.0
        # Relative gravity connectivity. Destination population cancels after row
        # normalization, but retaining the classical pairwise term makes the
        # construction explicit and easy to replace with observed flows later.
        raw = population[:, None] * population[None, :] / np.power(distance, distance_power)
        np.fill_diagonal(raw, 0.0)
        if top_k > 0 and top_k < len(units) - 1:
            keep = np.argpartition(raw, -top_k, axis=1)[:, -top_k:]
            mask = np.zeros_like(raw, dtype=bool)
            rows = np.arange(len(units))[:, None]
            mask[rows, keep] = True
            raw = np.where(mask, raw, 0.0)
        weights = raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)
        # row i receives pressure from source j.
        pressure = weights @ np.log1p(incidence)
        pressure_by_date.append(
            pd.DataFrame({"fcode": units, "date": date, "gravity_import_pressure": pressure})
        )
    pressure = pd.concat(pressure_by_date, ignore_index=True)
    return output.merge(pressure, on=["fcode", "date"], how="left", validate="one_to_one")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monthly", type=Path, required=True)
    parser.add_argument("--centroids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adjacency", type=Path)
    parser.add_argument("--distance-power", type=float, default=2.0)
    parser.add_argument("--minimum-distance-km", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    frame = pd.read_csv(args.monthly, dtype={"fcode": str})
    if "date" not in frame.columns:
        frame["date"] = pd.to_datetime({"year": frame["year"], "month": frame["month"], "day": 1})
    else:
        frame["date"] = pd.to_datetime(frame["date"])
    frame["population"] = pd.to_numeric(frame["population"], errors="coerce")
    frame = add_temporal_features(frame)
    frame = add_climate_anomalies(frame)
    if "l1_code" in frame.columns:
        frame = add_province_loo_incidence(frame)
    if args.adjacency is not None:
        adjacency = pd.read_csv(args.adjacency)
        frame = add_neighbor_features(frame, adjacency)
    centroids = pd.read_csv(args.centroids, dtype={"fcode": str})
    frame = add_gravity_import_pressure(
        frame,
        centroids,
        distance_power=args.distance_power,
        minimum_distance_km=args.minimum_distance_km,
        top_k=args.top_k,
    )
    # Preserve year/month based input format. Date is useful for inspection but
    # duplicated information, so omit it from the final panel.
    frame = frame.drop(columns=["date"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    added = sorted(set(frame.columns) - set(pd.read_csv(args.monthly, nrows=0).columns))
    print(f"wrote {args.output} with added features: {', '.join(added)}")


if __name__ == "__main__":
    main()
