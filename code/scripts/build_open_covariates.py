#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import time
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
ONI_URL = "https://psl.noaa.gov/data/correlation/oni.data"
DMI_URL = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GHSL_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss/"
    "V1-0/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip"
)
OPEN_DENGUE = (
    Path(__file__).resolve().parents[3]
    / "dengue_forecast_v1/dengue_training/data/external/opendengue/Temporal_extract_V1_3.csv"
)


def log(message: str) -> None:
    print(message, flush=True)


def fetch_bytes(url: str, timeout: int = 120) -> bytes:
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Failed {url}: {last_error}")


def fetch_json(url: str, timeout: int = 120) -> dict:
    return json.loads(fetch_bytes(url, timeout=timeout).decode("utf-8"))


def parse_psl_monthly(text: str, name: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        if year < 1900 or year > 2100:
            continue
        for month, token in enumerate(parts[1:13], start=1):
            value = float(token)
            if abs(value) >= 90:
                continue
            rows.append({"year": year, "month": month, name: value})
    if not rows:
        raise ValueError(f"Could not parse {name}")
    return pd.DataFrame(rows)


def unique_cells(centroids: pd.DataFrame, step: float) -> pd.DataFrame:
    frame = centroids.copy()
    frame["cell_lat"] = (frame["latitude"] / step).round() * step
    frame["cell_lon"] = (frame["longitude"] / step).round() * step
    cells = frame[["cell_lat", "cell_lon"]].drop_duplicates().reset_index(drop=True)
    cells["cell_id"] = np.arange(len(cells))
    mapped = frame.merge(cells, on=["cell_lat", "cell_lon"], how="left")
    return cells, mapped[["fcode", "cell_id"]]


def fetch_power_land(cells: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    frames = []
    api_versions = set()
    source_names = set()
    for row in cells.itertuples(index=False):
        params = (
            f"parameters=GWETTOP,ALLSKY_SFC_SW_DWN,T2MDEW&community=AG"
            f"&longitude={row.cell_lon:.4f}&latitude={row.cell_lat:.4f}"
            f"&start={start_year}&end={end_year}&format=JSON"
        )
        try:
            payload = fetch_json(f"{POWER_URL}?{params}")
        except Exception as exc:
            log(f"SKIP POWER cell {row.cell_id} ({row.cell_lat}, {row.cell_lon}): {exc}")
            continue
        parameters = payload["properties"]["parameter"]
        api_versions.add(str(payload.get("header", {}).get("api", {}).get("version", "unknown")))
        source_names.update(payload.get("header", {}).get("sources", []))
        dates = sorted(
            key
            for key in parameters["GWETTOP"]
            if not str(key).endswith("13")
        )
        part = pd.DataFrame(
            {
                "date": pd.to_datetime(dates, format="%Y%m"),
                "soil_moisture": [parameters["GWETTOP"][key] for key in dates],
                "solar_downward": [parameters["ALLSKY_SFC_SW_DWN"][key] for key in dates],
                "dewpoint_c": [parameters["T2MDEW"][key] for key in dates],
                "cell_id": row.cell_id,
            }
        )
        frames.append(part)
        log(f"POWER cell {int(row.cell_id) + 1}/{len(cells)}")
    if not frames:
        raise RuntimeError("NASA POWER returned no land-surface cells")
    result = pd.concat(frames, ignore_index=True)
    result.attrs["api_versions"] = sorted(api_versions)
    result.attrs["sources"] = sorted(source_names)
    return result


def fetch_et0(cells: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    frames = []
    for row in cells.itertuples(index=False):
        url = (
            f"{ARCHIVE_URL}?latitude={row.cell_lat:.4f}&longitude={row.cell_lon:.4f}"
            f"&start_date={start}&end_date={end}&daily=et0_fao_evapotranspiration"
        )
        payload = fetch_json(url)
        daily = pd.DataFrame(payload["daily"])
        daily["date"] = pd.to_datetime(daily["time"]).dt.to_period("M").dt.to_timestamp()
        monthly = (
            daily.groupby("date", as_index=False)["et0_fao_evapotranspiration"]
            .sum()
            .rename(columns={"et0_fao_evapotranspiration": "et0_mm"})
        )
        monthly["cell_id"] = row.cell_id
        frames.append(monthly)
        log(f"ET0 cell {int(row.cell_id) + 1}/{len(cells)}")
    return pd.concat(frames, ignore_index=True)


def fetch_elevation(centroids: pd.DataFrame) -> pd.DataFrame:
    latitudes = centroids["latitude"].to_numpy(float)
    longitudes = centroids["longitude"].to_numpy(float)
    values = np.empty(len(centroids), dtype=float)
    batch = 40
    for start in range(0, len(centroids), batch):
        stop = min(start + batch, len(centroids))
        lat = ",".join(f"{value:.5f}" for value in latitudes[start:stop])
        lon = ",".join(f"{value:.5f}" for value in longitudes[start:stop])
        payload = fetch_json(f"{ELEVATION_URL}?latitude={lat}&longitude={lon}")
        values[start:stop] = np.asarray(payload["elevation"], dtype=float)
    return pd.DataFrame({"fcode": centroids["fcode"].astype(str), "elevation_m": values})


def sample_ghsl(centroids: pd.DataFrame, cache_dir: Path) -> pd.DataFrame | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "ghsl_built_s_2020_4326_30ss.zip"
    if not archive.exists() or archive.stat().st_size < 1_000_000:
        log(f"downloading GHSL built-up ({GHSL_URL})")
        archive.write_bytes(fetch_bytes(GHSL_URL, timeout=600))
    try:
        import rasterio
    except ImportError:
        log("SKIP GHSL: rasterio is unavailable")
        return None
    tif_path = None
    with zipfile.ZipFile(archive) as handle:
        members = [name for name in handle.namelist() if name.lower().endswith(".tif")]
        if not members:
            log("SKIP GHSL: zip has no GeoTIFF")
            return None
        target = cache_dir / Path(members[0]).name
        if not target.exists():
            handle.extract(members[0], cache_dir)
            extracted = cache_dir / members[0]
            extracted.replace(target)
        tif_path = target
    points = list(zip(centroids["longitude"].to_numpy(float), centroids["latitude"].to_numpy(float)))
    with rasterio.open(tif_path) as src:
        samples = np.array([item[0] for item in src.sample(points)], dtype=float)
    samples = np.where(np.isfinite(samples) & (samples >= 0), samples, 0.0)
    return pd.DataFrame(
        {
            "fcode": centroids["fcode"].astype(str),
            "builtup_m2": samples,
            "log_builtup": np.log1p(samples),
        }
    )


def mekong_neighbor_dengue(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frames = []
    for chunk in pd.read_csv(
        path,
        usecols=["ISO_A0", "adm_1_name", "calendar_start_date", "dengue_total", "T_res"],
        chunksize=250_000,
        dtype={"ISO_A0": str},
    ):
        keep = chunk["T_res"].astype(str).eq("Month") & chunk["ISO_A0"].isin(["THA", "KHM", "LAO"])
        part = chunk.loc[keep].copy()
        if part.empty:
            continue
        part["date"] = pd.to_datetime(part["calendar_start_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        part["dengue_total"] = pd.to_numeric(part["dengue_total"], errors="coerce")
        part = part[part["date"].notna() & part["dengue_total"].notna()]
        frames.append(part)
    if not frames:
        raise ValueError("No Mekong OpenDengue monthly rows")
    table = pd.concat(frames, ignore_index=True)
    national = table[table["adm_1_name"].isna() | table["adm_1_name"].astype(str).eq("")]
    source = national if len(national) else table
    monthly = source.groupby(["ISO_A0", "date"], as_index=False)["dengue_total"].sum()
    wide = monthly.pivot(index="date", columns="ISO_A0", values="dengue_total").fillna(0.0)
    output = pd.DataFrame(
        {
            "date": wide.index,
            "neighbor_dengue_log": np.log1p(wide.sum(axis=1).to_numpy(float)),
        }
    )
    return output


def radiation_import_pressure(monthly: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    units = tuple(sorted(monthly["fcode"].astype(str).unique()))
    centroids = centroids.copy()
    centroids["fcode"] = centroids["fcode"].astype(str)
    centroids = centroids.set_index("fcode").reindex(units)
    lat = np.radians(centroids["latitude"].to_numpy(float))
    lon = np.radians(centroids["longitude"].to_numpy(float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    distance = 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    np.fill_diagonal(distance, 0.0)
    order = np.argsort(distance, axis=1)
    outputs = []
    for date, part in monthly.groupby("date", sort=True):
        indexed = part.set_index("fcode").reindex(units)
        population = indexed["population"].to_numpy(float)
        incidence = indexed["obs_dengue_cases"].to_numpy(float) / population * 100_000.0
        ranked_pop = population[order]
        self_pop = ranked_pop[:, :1]
        closer = np.maximum(np.cumsum(ranked_pop, axis=1) - ranked_pop - self_pop, 0.0)
        dest = population[:, None]
        origin = ranked_pop
        denom = np.maximum((dest + closer) * (dest + origin + closer), 1e-12)
        flow = dest * origin / denom
        flow[:, 0] = 0.0
        weights = np.zeros((len(units), len(units)), dtype=float)
        weights[np.arange(len(units))[:, None], order] = flow
        np.fill_diagonal(weights, 0.0)
        total = np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        pressure = (weights / total) @ np.log1p(incidence)
        outputs.append(
            pd.DataFrame({"fcode": units, "date": date, "radiation_import_pressure": pressure})
        )
    return pd.concat(outputs, ignore_index=True)


def mosquito_suitability(temp_k: np.ndarray) -> np.ndarray:
    temp_c = temp_k - 273.15
    tmin, tmax, scale = 17.05, 34.61, 2.7e-4
    value = np.zeros_like(temp_c, dtype=float)
    ok = (temp_c > tmin) & (temp_c < tmax)
    value[ok] = scale * temp_c[ok] * (temp_c[ok] - tmin) * np.sqrt(tmax - temp_c[ok])
    peak = scale * 29.0 * (29.0 - tmin) * math.sqrt(tmax - 29.0)
    return value / max(peak, 1e-12)


def climatology_forecasts(monthly: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    frame = monthly.sort_values(["fcode", "date"]).copy()
    rows = []
    climate_map = {"t2m_avg": "t2m_forecast", "rh_avg": "rh_forecast", "tp_accum": "tp_forecast"}
    for fcode, part in frame.groupby("fcode", sort=False):
        part = part.reset_index(drop=True)
        years = part["date"].dt.year.to_numpy()
        months = part["date"].dt.month.to_numpy()
        for issue_index in range(len(part)):
            for horizon in horizons:
                target_index = issue_index + horizon
                if target_index >= len(part):
                    continue
                history = (years < years[issue_index]) & (months == months[target_index])
                row = {
                    "fcode": str(fcode),
                    "issue_date": part.at[issue_index, "date"],
                    "target_date": part.at[target_index, "date"],
                }
                for source, destination in climate_map.items():
                    if not history.any():
                        row[destination] = float(part.at[issue_index, source])
                    else:
                        row[destination] = float(part.loc[history, source].mean())
                rows.append(row)
    return pd.DataFrame(rows)


def causal_fill(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.sort_values(["fcode", "date"]).copy()
    for column in columns:
        # Forward propagation is admissible at an issue date; backward filling
        # would copy a future observation into an earlier month and leak data.
        output[column] = output.groupby("fcode", sort=False)[column].ffill()
        output[column] = output[column].fillna(0.0)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monthly", type=Path, required=True)
    parser.add_argument("--centroids", type=Path, required=True)
    parser.add_argument("--output-monthly", type=Path, required=True)
    parser.add_argument("--output-future", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    monthly = pd.read_csv(args.monthly, dtype={"fcode": str, "l1_code": str})
    if "date" not in monthly.columns:
        monthly["date"] = pd.to_datetime({"year": monthly["year"], "month": monthly["month"], "day": 1})
    else:
        monthly["date"] = pd.to_datetime(monthly["date"])
    centroids = pd.read_csv(args.centroids, dtype={"fcode": str})
    start_year = int(monthly["date"].dt.year.min())
    end_year = min(int(monthly["date"].dt.year.max()), 2025)
    notes = {}

    oni = parse_psl_monthly(fetch_bytes(ONI_URL).decode("utf-8"), "nino34_index")
    dmi = parse_psl_monthly(fetch_bytes(DMI_URL).decode("utf-8"), "iod_dmi")
    climate_idx = oni.merge(dmi, on=["year", "month"], how="outer")
    climate_idx["date"] = pd.to_datetime({"year": climate_idx["year"], "month": climate_idx["month"], "day": 1})
    climate_idx = climate_idx.sort_values("date")
    climate_idx["nino34_lag2"] = climate_idx["nino34_index"].shift(2)
    climate_idx["iod_dmi_lag1"] = climate_idx["iod_dmi"].shift(1)
    notes["enso"] = "NOAA PSL ONI lag-2 and DMI lag-1"

    cells, cell_map = unique_cells(centroids, 0.5)
    try:
        land = fetch_power_land(cells, start_year, end_year)
        power_metadata = dict(land.attrs)
        land = cell_map.merge(land, on="cell_id", how="left")
        notes["nasa_power"] = {
            "variables": "GWETTOP, ALLSKY_SFC_SW_DWN, T2MDEW nearest 0.5-degree cell",
            "api_versions": power_metadata.get("api_versions", []),
            "sources": power_metadata.get("sources", []),
            "accessed_on": date.today().isoformat(),
        }
    except Exception as exc:
        log(f"SKIP NASA POWER: {exc}")
        land = None
        notes["nasa_power"] = f"unavailable: {exc}"

    try:
        et0 = fetch_et0(cells, f"{start_year}-01-01", f"{end_year}-12-31")
        et0 = cell_map.merge(et0, on="cell_id", how="left")
        notes["et0"] = {
            "description": "Open-Meteo ERA5-Land FAO ET0 monthly sum",
            "accessed_on": date.today().isoformat(),
        }
    except Exception as exc:
        log(f"SKIP ET0: {exc}")
        et0 = None
        notes["et0"] = f"unavailable: {exc}"

    elevation = fetch_elevation(centroids)
    notes["elevation"] = "Open-Meteo elevation at commune centroids"

    try:
        built = sample_ghsl(centroids, args.cache_dir / "ghsl")
        notes["builtup"] = "GHSL GHS-BUILT-S 2020 30 arc-second WGS84, centroid sample"
    except Exception as exc:
        log(f"SKIP GHSL: {exc}")
        built = None
        notes["builtup"] = f"unavailable: {exc}"

    try:
        neighbor = mekong_neighbor_dengue(OPEN_DENGUE)
        notes["opendengue"] = "OpenDengue monthly THA+KHM+LAO log1p total"
    except Exception as exc:
        log(f"SKIP OpenDengue: {exc}")
        neighbor = None
        notes["opendengue"] = f"unavailable: {exc}"

    radiation = radiation_import_pressure(monthly, centroids)
    notes["radiation_mobility"] = "Simini radiation-model import pressure from population and distance"

    panel = monthly.copy()
    panel["mosquito_suitability"] = mosquito_suitability(panel["t2m_avg"].to_numpy(float))
    panel = panel.merge(climate_idx[["date", "nino34_lag2", "iod_dmi_lag1"]], on="date", how="left")
    if land is not None:
        panel = panel.merge(
            land[["fcode", "date", "soil_moisture", "solar_downward", "dewpoint_c"]],
            on=["fcode", "date"],
            how="left",
        )
    if et0 is not None:
        panel = panel.merge(et0[["fcode", "date", "et0_mm"]], on=["fcode", "date"], how="left")
    panel = panel.merge(elevation, on="fcode", how="left")
    if built is not None:
        panel = panel.merge(built, on="fcode", how="left")
    if neighbor is not None:
        panel = panel.merge(neighbor, on="date", how="left")
    panel = panel.merge(radiation, on=["fcode", "date"], how="left")
    fill_columns = [
        column
        for column in (
            "nino34_lag2",
            "iod_dmi_lag1",
            "soil_moisture",
            "solar_downward",
            "dewpoint_c",
            "et0_mm",
            "elevation_m",
            "builtup_m2",
            "log_builtup",
            "neighbor_dengue_log",
            "radiation_import_pressure",
            "mosquito_suitability",
        )
        if column in panel.columns
    ]
    panel = causal_fill(panel, fill_columns)
    panel = panel.drop(columns=["date"])
    args.output_monthly.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output_monthly, index=False)

    future = climatology_forecasts(monthly, (1, 2, 3))
    args.output_future.parent.mkdir(parents=True, exist_ok=True)
    future.to_csv(args.output_future, index=False)
    metadata = {
        "leakage_rule": "every value is known at or before the commune-month / issue_date",
        "future_climate": "expanding same-calendar-month climatology from years strictly before issue year",
        "notes": notes,
        "columns": fill_columns,
        "rows": int(len(panel)),
    }
    (args.output_monthly.with_suffix(".metadata.json")).write_text(json.dumps(metadata, indent=2) + "\n")
    log(f"wrote {args.output_monthly} and {args.output_future}")
    log("added: " + ", ".join(fill_columns))


if __name__ == "__main__":
    main()
