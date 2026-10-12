from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SnapshotData:
    origins: pd.DatetimeIndex
    units: tuple[str, ...]
    provinces: tuple[str, ...]
    horizons: tuple[int, ...]
    local: np.ndarray
    climate: np.ndarray
    seasonal: np.ndarray
    targets: np.ndarray
    population: np.ndarray
    target_dates: np.ndarray
    outbreak_thresholds: np.ndarray

    def subset(self, mask: np.ndarray) -> "SnapshotData":
        return SnapshotData(
            origins=self.origins[mask],
            units=self.units,
            provinces=self.provinces,
            horizons=self.horizons,
            local=self.local[mask],
            climate=self.climate[mask],
            seasonal=self.seasonal[mask],
            targets=self.targets[mask],
            population=self.population[mask],
            target_dates=self.target_dates[mask],
            outbreak_thresholds=self.outbreak_thresholds[mask],
        )


def load_monthly(path: str | Path, config: dict) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={config["unit_column"]: str})
    history_covariates = list(config.get("history_covariate_columns", []))
    required = {
        config["unit_column"],
        config["province_column"],
        "year",
        "month",
        config["case_column"],
        config["population_column"],
        *config["climate_columns"],
        *history_covariates,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Monthly data are missing columns: {missing}")
    frame = frame.rename(
        columns={
            config["unit_column"]: "fcode",
            config["province_column"]: "province",
            config["case_column"]: "cases",
            config["population_column"]: "population",
        }
    )
    frame["fcode"] = frame["fcode"].astype(str)
    frame["province"] = frame["province"].astype(str)
    frame["date"] = pd.to_datetime(
        {"year": frame["year"], "month": frame["month"], "day": 1}
    )
    frame["cases"] = pd.to_numeric(frame["cases"], errors="coerce")
    frame["population"] = pd.to_numeric(frame["population"], errors="coerce")
    if frame.duplicated(["fcode", "date"]).any():
        raise ValueError("Monthly data contain duplicate commune-month rows")
    if frame["cases"].isna().any() or (frame["cases"] < 0).any():
        raise ValueError("Cases must be observed non-negative counts")
    if frame["population"].isna().any() or (frame["population"] <= 0).any():
        raise ValueError("Population must be positive")
    for column in [*config["climate_columns"], *history_covariates]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"Covariate column {column!r} contains missing values")
    return frame.sort_values(["date", "fcode"]).reset_index(drop=True)


def load_future_covariates(
    path: str | Path,
    columns: list[str] | tuple[str, ...],
    *,
    unit_column: str = "fcode",
    issue_column: str = "issue_date",
    target_column: str = "target_date",
) -> pd.DataFrame:
    """Load issue-time future covariates without leaking realised future values.

    Expected one row per commune, issue month and target month. Values must be
    forecasts or otherwise information genuinely available at ``issue_date``.
    """
    frame = pd.read_csv(path, dtype={unit_column: str})
    required = {unit_column, issue_column, target_column, *columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Future-covariate data are missing columns: {missing}")
    frame = frame.rename(
        columns={
            unit_column: "fcode",
            issue_column: "issue_date",
            target_column: "target_date",
        }
    )
    frame["fcode"] = frame["fcode"].astype(str)
    frame["issue_date"] = pd.to_datetime(frame["issue_date"]).dt.to_period("M").dt.to_timestamp()
    frame["target_date"] = pd.to_datetime(frame["target_date"]).dt.to_period("M").dt.to_timestamp()
    if (frame["target_date"] <= frame["issue_date"]).any():
        raise ValueError("Future covariates must have target_date after issue_date")
    if frame.duplicated(["fcode", "issue_date", "target_date"]).any():
        raise ValueError("Future-covariate data contain duplicate fcode/issue/target rows")
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"Future covariate {column!r} contains missing values")
    return frame.sort_values(["issue_date", "target_date", "fcode"]).reset_index(drop=True)


def load_outbreak_thresholds(
    path: str | Path,
    *,
    unit_column: str = "fcode",
    threshold_column: str = "epi_threshold",
) -> pd.DataFrame:
    """Load externally supplied commune- and target-month outbreak thresholds."""
    frame = pd.read_csv(path, dtype={unit_column: str})
    required = {unit_column, "year", "month", threshold_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Outbreak-threshold data are missing columns: {missing}")
    frame = frame.rename(
        columns={unit_column: "fcode", threshold_column: "outbreak_threshold"}
    )
    frame["fcode"] = frame["fcode"].astype(str)
    frame["date"] = pd.to_datetime(
        {"year": frame["year"], "month": frame["month"], "day": 1}
    )
    frame["outbreak_threshold"] = pd.to_numeric(
        frame["outbreak_threshold"], errors="coerce"
    )
    if frame.duplicated(["fcode", "date"]).any():
        raise ValueError("Outbreak-threshold data contain duplicate commune-month rows")
    if (frame["outbreak_threshold"].dropna() < 0).any():
        raise ValueError("Observed outbreak thresholds must be non-negative")
    # Early target months may not have the five eligible historical years
    # required by the MOH definition.  Keep those rows out of the lookup; the
    # snapshot builder represents unavailable labels as infinity and excludes
    # them from outbreak loss/metrics.
    frame = frame.dropna(subset=["outbreak_threshold"])
    return frame[["fcode", "date", "outbreak_threshold"]].sort_values(
        ["date", "fcode"]
    ).reset_index(drop=True)


def validate_complete_panel(frame: pd.DataFrame) -> None:
    dates = pd.date_range(frame["date"].min(), frame["date"].max(), freq="MS")
    units = frame["fcode"].drop_duplicates()
    expected = len(dates) * len(units)
    if len(frame) != expected:
        observed = pd.MultiIndex.from_frame(frame[["date", "fcode"]])
        complete = pd.MultiIndex.from_product([dates, units], names=["date", "fcode"])
        examples = complete.difference(observed).tolist()[:10]
        raise ValueError(
            f"Panel has {len(frame)} rows but {expected} are required; "
            f"example missing rows: {examples}"
        )


def build_snapshots(
    frame: pd.DataFrame,
    config: dict,
    future_covariates: pd.DataFrame | None = None,
    outbreak_thresholds: pd.DataFrame | None = None,
) -> SnapshotData:
    validate_complete_panel(frame)
    horizons = tuple(int(value) for value in config["forecast"]["horizons"])
    lookback = int(config["forecast"]["sequence_length"])
    lag_years = tuple(int(value) for value in config["forecast"]["seasonal_lag_years"])
    climate_columns = list(config["data"]["climate_columns"])
    history_covariates = list(config["data"].get("history_covariate_columns", []))
    future_columns = list(config["data"].get("future_covariate_columns", []))
    if future_columns and future_covariates is None:
        raise ValueError(
            "future_covariate_columns are configured but no future-covariate table was loaded"
        )
    units = tuple(sorted(frame["fcode"].unique()))
    province_lookup = (
        frame[["fcode", "province"]].drop_duplicates().set_index("fcode")["province"]
    )
    if province_lookup.index.duplicated().any():
        raise ValueError("A commune maps to more than one province in the study panel")
    provinces = tuple(province_lookup.loc[list(units)].astype(str))
    dates = pd.date_range(frame["date"].min(), frame["date"].max(), freq="MS")

    cases = _pivot(frame, dates, units, "cases")
    population = _pivot(frame, dates, units, "population")
    incidence = cases / population * 100_000.0
    month = np.asarray([date.month for date in dates], dtype=np.float32)
    calendar = np.stack(
        [
            np.sin(2.0 * np.pi * month / 12.0),
            np.cos(2.0 * np.pi * month / 12.0),
        ],
        axis=-1,
    )
    local_parts = [
        np.log1p(cases)[..., None],
        np.log1p(incidence)[..., None],
        np.log1p(population)[..., None],
        np.repeat(calendar[:, None, :], len(units), axis=1),
    ]
    if history_covariates:
        history = np.stack(
            [_pivot(frame, dates, units, column) for column in history_covariates],
            axis=-1,
        )
        local_parts.append(history)
    local = np.concatenate(local_parts, axis=-1)
    climate = np.stack(
        [_pivot(frame, dates, units, column) for column in climate_columns],
        axis=-1,
    )

    future_lookup = _future_covariate_lookup(future_covariates, future_columns, units)
    # Always construct the full MOH definition from the model's case history.
    # An audited export, when supplied, overrides matching rows.  This makes a
    # missing file or a few missing early rows a row-level fallback rather than
    # silently switching the whole analysis to a different definition.
    threshold_lookup = _moh_outbreak_threshold_lookup(frame, dates, units)
    threshold_lookup.update(_outbreak_threshold_lookup(outbreak_thresholds, units))
    max_horizon = max(horizons)
    first_origin = lookback - 1
    origin_indices = range(first_origin, len(dates) - max_horizon)
    local_values: list[np.ndarray] = []
    climate_values: list[np.ndarray] = []
    seasonal_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    population_values: list[np.ndarray] = []
    target_date_values: list[np.ndarray] = []
    outbreak_threshold_values: list[np.ndarray] = []
    origins: list[pd.Timestamp] = []

    for origin in origin_indices:
        local_values.append(
            local[origin - lookback + 1 : origin + 1].transpose(1, 0, 2)
        )
        climate_values.append(
            climate[origin - lookback + 1 : origin + 1].transpose(1, 0, 2)
        )
        seasonal = _target_aligned_history(cases, origin, horizons, lag_years)
        thresholds = _outbreak_thresholds_for_targets(
            dates, origin, horizons, units, threshold_lookup
        )
        if future_columns:
            future = _future_context_for_origin(
                dates[origin], dates, origin, horizons, units, future_lookup, future_columns
            )
            seasonal = np.concatenate([seasonal, future], axis=-1)
        seasonal_values.append(seasonal)
        outbreak_threshold_values.append(thresholds)
        target_values.append(
            np.stack([cases[origin + horizon] for horizon in horizons], axis=-1)
        )
        population_values.append(
            np.stack([population[origin + horizon] for horizon in horizons], axis=-1)
        )
        target_date_values.append(
            np.asarray([dates[origin + horizon] for horizon in horizons])
        )
        origins.append(dates[origin])

    return SnapshotData(
        origins=pd.DatetimeIndex(origins),
        units=units,
        provinces=provinces,
        horizons=horizons,
        local=np.stack(local_values).astype(np.float32),
        climate=np.stack(climate_values).astype(np.float32),
        seasonal=np.stack(seasonal_values).astype(np.float32),
        targets=np.stack(target_values).astype(np.float32),
        population=np.stack(population_values).astype(np.float32),
        target_dates=np.stack(target_date_values),
        outbreak_thresholds=np.stack(outbreak_threshold_values).astype(np.float32),
    )


def split_mask(data: SnapshotData, start: str, end: str) -> np.ndarray:
    # Multi-horizon models share issue months; use the longest target as the
    # split anchor so no origin is assigned to different splits by horizon.
    anchor = data.horizons.index(max(data.horizons))
    target_dates = pd.to_datetime(data.target_dates[:, anchor])
    return np.asarray(
        (target_dates >= pd.Timestamp(start)) & (target_dates <= pd.Timestamp(end))
    )


def _pivot(
    frame: pd.DataFrame,
    dates: pd.DatetimeIndex,
    units: tuple[str, ...],
    value: str,
) -> np.ndarray:
    return (
        frame.pivot(index="date", columns="fcode", values=value)
        .reindex(index=dates, columns=units)
        .to_numpy(dtype=np.float32)
    )


def _target_aligned_history(
    cases: np.ndarray,
    origin: int,
    horizons: tuple[int, ...],
    lag_years: tuple[int, ...],
) -> np.ndarray:
    by_horizon = []
    for horizon in horizons:
        target = origin + horizon
        raw = np.full((cases.shape[1], len(lag_years)), np.nan, dtype=np.float32)
        for position, lag in enumerate(lag_years):
            historical_index = target - 12 * lag
            if historical_index >= 0:
                raw[:, position] = cases[historical_index]
        available = np.isfinite(raw)
        logged = np.zeros_like(raw)
        logged[available] = np.log1p(raw[available])
        counts = available.sum(axis=-1, keepdims=True).clip(min=1)
        mean = logged.sum(axis=-1, keepdims=True) / counts
        centred = np.where(available, logged - mean, 0.0)
        sd = np.sqrt(
            np.square(centred).sum(axis=-1, keepdims=True)
            / np.maximum(counts - 1, 1)
        )
        newest = np.argmax(available, axis=-1)
        oldest = available.shape[1] - 1 - np.argmax(available[:, ::-1], axis=-1)
        row = np.arange(len(raw))
        trend = (logged[row, newest] - logged[row, oldest])[:, None]
        zero_share = (
            np.where(available, raw == 0, 0).sum(axis=-1, keepdims=True) / counts
        )
        by_horizon.append(
            np.concatenate(
                [logged, available.astype(np.float32), mean, sd, trend, zero_share],
                axis=-1,
            )
        )
    return np.stack(by_horizon, axis=1)


def _outbreak_threshold_lookup(
    frame: pd.DataFrame | None,
    units: tuple[str, ...],
) -> dict[tuple[str, pd.Timestamp], float]:
    if frame is None:
        return {}
    relevant = frame[frame["fcode"].isin(units)]
    return {
        (row.fcode, pd.Timestamp(row.date)): float(row.outbreak_threshold)
        for row in relevant.itertuples(index=False)
    }


def _moh_outbreak_threshold_lookup(
    frame: pd.DataFrame,
    dates: pd.DatetimeIndex,
    units: tuple[str, ...],
) -> dict[tuple[str, pd.Timestamp], float]:
    """Compute MOH thresholds, with an early-history fallback per missing row.

    The primary calculation uses the five latest same-calendar-month years,
    tests each candidate against mean + 2 sample SD of the other four, removes
    the strongest outbreak candidate, and replaces it with the next older year.
    When five usable reference years do not exist, the old mean + 2 SD rule is
    used only for that early row (and only when at least two years are present).
    """
    history: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for (fcode, month), group in frame.groupby(
        ["fcode", frame["date"].dt.month], sort=False
    ):
        history[(str(fcode), int(month))] = sorted(
            (
                (int(row.date.year), float(row.cases))
                for row in group[["date", "cases"]].itertuples(index=False)
                if math.isfinite(float(row.cases))
            ),
            reverse=True,
        )

    output: dict[tuple[str, pd.Timestamp], float] = {}
    for unit in units:
        for date in dates:
            candidates = [
                (year, value)
                for year, value in history.get((unit, int(date.month)), [])
                if int(date.year) - 15 <= year < int(date.year)
            ]
            selected = candidates[:5]
            remaining = candidates[5:]
            if len(selected) == 5:
                selected = _exclude_moh_outbreak_year(selected, remaining)
            if len(selected) >= 5:
                values = np.asarray([value for _, value in selected], dtype=float)
                threshold = float(values.mean() + 2.0 * values.std(ddof=1))
            elif len(candidates) >= 2:
                # Explicitly limited to the genuinely under-observed early era.
                values = np.asarray([value for _, value in candidates[:5]], dtype=float)
                threshold = float(values.mean() + 2.0 * values.std(ddof=1))
            else:
                continue
            output[(unit, pd.Timestamp(date))] = threshold
    return output


def _exclude_moh_outbreak_year(
    selected: list[tuple[int, float]],
    remaining: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Apply the same candidate-year exclusion rule as the audited MOH code."""
    values = [value for _, value in selected]
    total = sum(values)
    total_sq = sum(value * value for value in values)
    exceedances: list[tuple[float, int, int]] = []
    for index, (year, value) in enumerate(selected):
        other_mean = (total - value) / 4
        centered_sum_sq = (total_sq - value * value) - 4 * other_mean * other_mean
        other_sd = math.sqrt(max(centered_sum_sq / 3, 0.0))
        threshold = other_mean + 2.0 * other_sd
        if value > threshold:
            exceedances.append((value - threshold, year, index))
    if not exceedances:
        return list(selected)
    _, _, index = max(exceedances, key=lambda item: (item[0], item[1]))
    output = list(selected)
    output.pop(index)
    if remaining:
        output.append(remaining[0])
        output.sort(key=lambda item: item[0], reverse=True)
    return output


def _outbreak_thresholds_for_targets(
    dates: pd.DatetimeIndex,
    origin: int,
    horizons: tuple[int, ...],
    units: tuple[str, ...],
    lookup: dict[tuple[str, pd.Timestamp], float],
) -> np.ndarray:
    # The very first years can still be unavailable when fewer than two prior
    # same-month observations exist; infinity excludes only those labels.
    values = np.full((len(units), len(horizons)), np.inf, dtype=np.float32)
    for node, unit in enumerate(units):
        for horizon_index, horizon in enumerate(horizons):
            value = lookup.get((unit, pd.Timestamp(dates[origin + horizon])))
            if value is not None:
                values[node, horizon_index] = value
    return values


def _future_covariate_lookup(
    frame: pd.DataFrame | None,
    columns: list[str],
    units: tuple[str, ...],
) -> dict[tuple[str, pd.Timestamp, pd.Timestamp], np.ndarray]:
    if not columns or frame is None:
        return {}
    unknown = sorted(set(frame["fcode"]) - set(units))
    if unknown:
        frame = frame[frame["fcode"].isin(units)].copy()
    return {
        (row.fcode, row.issue_date, row.target_date): np.asarray(
            [getattr(row, column) for column in columns], dtype=np.float32
        )
        for row in frame.itertuples(index=False)
    }


def _future_context_for_origin(
    issue_date: pd.Timestamp,
    dates: pd.DatetimeIndex,
    origin: int,
    horizons: tuple[int, ...],
    units: tuple[str, ...],
    lookup: dict[tuple[str, pd.Timestamp, pd.Timestamp], np.ndarray],
    columns: list[str],
) -> np.ndarray:
    values = np.empty((len(units), len(horizons), len(columns)), dtype=np.float32)
    missing: list[tuple[str, str, str]] = []
    for node, unit in enumerate(units):
        for h_index, horizon in enumerate(horizons):
            target_date = pd.Timestamp(dates[origin + horizon])
            key = (unit, pd.Timestamp(issue_date), target_date)
            if key not in lookup:
                missing.append((unit, str(issue_date.date()), str(target_date.date())))
                continue
            values[node, h_index] = lookup[key]
    if missing:
        example = missing[:5]
        raise ValueError(
            "Future covariates do not cover every snapshot; examples "
            f"(fcode, issue_date, target_date): {example}"
        )
    return values
