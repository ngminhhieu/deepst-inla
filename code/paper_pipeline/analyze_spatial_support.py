#!/usr/bin/env python3
"""Controlled nested aggregation experiment for the spatial-support mechanism.

No model is retrained.  Commune-level predictive draws, observations, and
populations are summed over deterministic nested spatial groups.  Consequently
the only manipulated factor is spatial support (plus the unavoidable change in
event prevalence induced by the official threshold definition).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

mpl.rcParams.update({
    "font.size": 12.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 10.5,
})

from dengue_h3.evaluation import _crps  # noqa: E402
from analyze_province_scale import (  # noqa: E402
    apply_platt,
    calibration_diagnostic,
    classification,
    fit_platt,
    threshold_for_constraint,
)


def allocate(total: int, sizes: dict[str, int]) -> dict[str, int]:
    """Largest-remainder allocation with at least one group per province."""
    if total < len(sizes):
        raise ValueError("Target group count is smaller than province count")
    n = sum(sizes.values())
    raw = {key: total * value / n for key, value in sizes.items()}
    result = {key: max(1, min(sizes[key], int(np.floor(value)))) for key, value in raw.items()}
    while sum(result.values()) < total:
        eligible = [key for key in sizes if result[key] < sizes[key]]
        key = max(eligible, key=lambda item: (raw[item] - result[item], sizes[item], item))
        result[key] += 1
    while sum(result.values()) > total:
        eligible = [key for key in sizes if result[key] > 1]
        key = min(eligible, key=lambda item: (raw[item] - result[item], -sizes[item], item))
        result[key] -= 1
    return result


def cluster_once(
    current: pd.DataFrame, edges: set[tuple[str, str]], target: int, level: str
) -> pd.DataFrame:
    """Merge current groups within provinces; never split a prior group."""
    sizes = current.groupby("l1_code").size().astype(int).to_dict()
    targets = allocate(target, sizes)
    output = []
    for province, part in current.groupby("l1_code", sort=True):
        part = part.reset_index(drop=True)
        k = targets[str(province)]
        if k == len(part):
            labels = np.arange(k)
        elif k == 1:
            labels = np.zeros(len(part), dtype=int)
        else:
            index = {value: i for i, value in enumerate(part.group_id)}
            rows, columns = [], []
            for left, right in edges:
                if left in index and right in index:
                    rows.extend((index[left], index[right]))
                    columns.extend((index[right], index[left]))
            connectivity = sparse.csr_matrix(
                (np.ones(len(rows)), (rows, columns)), shape=(len(part), len(part))
            ).tolil()
            # Administrative graphs contain islands. Connect components by the
            # shortest centroid bridge so the spatial constraint is explicit
            # and deterministic instead of letting sklearn complete it opaquely.
            while True:
                component_count, component = sparse.csgraph.connected_components(
                    connectivity, directed=False
                )
                if component_count == 1:
                    break
                coordinates = part[["longitude", "latitude"]].to_numpy(float)
                best = None
                for left in range(len(part)):
                    for right in range(left + 1, len(part)):
                        if component[left] == component[right]:
                            continue
                        distance = float(np.square(coordinates[left] - coordinates[right]).sum())
                        candidate = (distance, left, right)
                        if best is None or candidate < best:
                            best = candidate
                _, left, right = best
                connectivity[left, right] = connectivity[right, left] = 1
            labels = AgglomerativeClustering(
                n_clusters=k, linkage="ward", connectivity=connectivity.tocsr()
            ).fit_predict(part[["longitude", "latitude"]].to_numpy(float))
        part["parent"] = [f"{level}_{province}_{value:03d}" for value in labels]
        output.append(part)
    membership = pd.concat(output, ignore_index=True)
    next_groups = membership.groupby(["parent", "l1_code"], as_index=False).agg(
        longitude=("longitude", "mean"), latitude=("latitude", "mean")
    ).rename(columns={"parent": "group_id"})
    edge_frame = membership[["group_id", "parent"]].drop_duplicates()
    parent_of = dict(edge_frame.itertuples(index=False, name=None))
    next_edges = {
        tuple(sorted((parent_of[left], parent_of[right])))
        for left, right in edges
        if left in parent_of and right in parent_of and parent_of[left] != parent_of[right]
    }
    return membership[["group_id", "parent"]], next_groups, next_edges


def nested_maps(
    units: list[str], monthly: pd.DataFrame, centroids: pd.DataFrame,
    adjacency: pd.DataFrame, targets: tuple[int, ...],
) -> dict[str, pd.DataFrame]:
    unit_info = (
        monthly[monthly.fcode.isin(units)][["fcode", "l1_code"]]
        .drop_duplicates("fcode").merge(centroids, on="fcode", how="left")
    )
    if len(unit_info) != len(units) or unit_info[["longitude", "latitude"]].isna().any().any():
        raise ValueError("Incomplete unit metadata")
    current = unit_info.rename(columns={"fcode": "group_id"}).copy()
    edges = {
        tuple(sorted((str(row[0]), str(row[1]))))
        for row in adjacency[["from", "to"]].itertuples(index=False, name=None)
        if str(row[0]) in units and str(row[1]) in units
    }
    original_to_current = pd.DataFrame({"fcode": units, "group_id": units})
    maps = {"commune_554": original_to_current.rename(columns={"group_id": "aggregate_id"})}
    for target in targets:
        membership, current, edges = cluster_once(current, edges, target, f"support_{target}")
        parent = dict(membership.itertuples(index=False, name=None))
        original_to_current["group_id"] = original_to_current.group_id.map(parent)
        maps[f"support_{target}"] = original_to_current.rename(
            columns={"group_id": "aggregate_id"}
        ).copy()
    province = unit_info[["fcode", "l1_code"]].rename(columns={"l1_code": "aggregate_id"})
    maps[f"province_{province.aggregate_id.nunique()}"] = province
    return maps


def aggregate_history(monthly: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    work = monthly.merge(mapping, on="fcode", how="inner")
    work["date"] = pd.to_datetime(dict(year=work.year, month=work.month, day=1))
    return work.groupby(["aggregate_id", "date"], as_index=False).agg(
        obs_dengue_cases=("obs_dengue_cases", "sum"), population=("population", "sum")
    ).rename(columns={"aggregate_id": "fcode"})


def build_primary_thresholds(
    history: pd.DataFrame, target_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Primary study threshold: preceding five same-month values, mean + 2 s.d."""
    data = history.copy()
    data["date"] = pd.to_datetime(data.date)
    data["year"] = data.date.dt.year
    data["month"] = data.date.dt.month
    rows = []
    for (unit, month), part in data.groupby(["fcode", "month"], sort=False):
        series = part.set_index("year").obs_dengue_cases
        for target_date in target_dates[target_dates.month == month]:
            values = series[
                (series.index < target_date.year)
                & (series.index >= target_date.year - 5)
            ].to_numpy(float)
            threshold = (
                float(values.mean() + 2 * values.std(ddof=1))
                if len(values) >= 2 else np.nan
            )
            rows.append({
                "fcode": str(unit), "date": target_date,
                "moh_threshold_2sd_cases": threshold,
            })
    return pd.DataFrame(rows)


def aggregate_split(
    frame: pd.DataFrame, cube: np.ndarray, mapping: pd.DataFrame,
    thresholds: pd.DataFrame, level: str,
) -> tuple[pd.DataFrame, dict[tuple[int, str], np.ndarray]]:
    units = list(frame[frame.horizon.eq(1)].fcode.drop_duplicates())
    unit_index = {unit: index for index, unit in enumerate(units)}
    groups = {
        group: np.array([unit_index[value] for value in part.fcode], dtype=int)
        for group, part in mapping.groupby("aggregate_id", sort=True)
    }
    threshold_lookup = {
        (str(row.fcode), pd.Timestamp(row.date)): float(row.moh_threshold_2sd_cases)
        for row in thresholds.itertuples(index=False)
    }
    rows, aggregated_samples = [], {}
    horizons = sorted(frame.horizon.unique())
    for horizon_index, horizon in enumerate(horizons):
        part = frame[frame.horizon.eq(horizon)].reset_index(drop=True)
        dates = list(pd.to_datetime(part.target_date.drop_duplicates()))
        expected_units = list(part.iloc[: len(units)].fcode)
        if expected_units != units or len(part) != len(dates) * len(units):
            raise ValueError("Prediction ordering is not date-major / unit-minor")
        observed = part.target_cases.to_numpy(float).reshape(len(dates), len(units))
        population = part.population.to_numpy(float).reshape(len(dates), len(units))
        values = cube[:, :, horizon_index, :]
        if values.shape[:2] != observed.shape:
            raise ValueError("Predictive cube and rows do not align")
        for group, indices in groups.items():
            sample_sum = values[:, indices, :].sum(axis=1)
            aggregated_samples[(int(horizon), str(group))] = sample_sum
            obs_sum = observed[:, indices].sum(axis=1)
            pop_sum = population[:, indices].sum(axis=1)
            for time_index, target_date in enumerate(dates):
                threshold = threshold_lookup.get((str(group), target_date), np.nan)
                if not np.isfinite(threshold):
                    raise ValueError(f"Missing threshold for {group} {target_date:%Y-%m}")
                rows.append({
                    "scale": level, "aggregate_id": str(group),
                    "communes_per_unit": len(indices), "horizon": int(horizon),
                    "target_date": target_date, "target_cases": obs_sum[time_index],
                    "population": pop_sum[time_index], "outbreak_threshold": threshold,
                    "raw_outbreak_probability": float(
                        np.mean(sample_sum[time_index] > threshold)
                    ),
                    "predicted_mean": float(sample_sum[time_index].mean()),
                })
    return pd.DataFrame(rows), aggregated_samples


def score_levels(validation: pd.DataFrame, test: pd.DataFrame, sample_store):
    metrics, matched, calibration, policies = [], [], [], []
    calibrated = []
    for scale in test.scale.drop_duplicates():
        for horizon in (1, 2, 3):
            valid = validation[(validation.scale == scale) & validation.horizon.eq(horizon)].copy()
            target = test[(test.scale == scale) & test.horizon.eq(horizon)].copy()
            vy = (valid.target_cases > valid.outbreak_threshold).to_numpy(bool)
            ty = (target.target_cases > target.outbreak_threshold).to_numpy(bool)
            intercept, slope = fit_platt(valid.raw_outbreak_probability, vy)
            vp = apply_platt(valid.raw_outbreak_probability, intercept, slope)
            tp = apply_platt(target.raw_outbreak_probability, intercept, slope)
            valid["outbreak_probability"] = vp
            target["outbreak_probability"] = tp
            valid["observed_outbreak"] = vy.astype(int)
            target["observed_outbreak"] = ty.astype(int)
            policies.append({
                "scale": scale, "horizon": horizon,
                "validation_platt_intercept": intercept,
                "validation_platt_slope": slope,
            })
            diag_i, diag_s, ece = calibration_diagnostic(ty, tp)
            calibration.append({
                "scale": scale, "horizon": horizon, "n": len(target),
                "test_calibration_intercept": diag_i,
                "test_calibration_slope": diag_s,
                "ece_5_equal_count_bins": ece,
            })
            baseline_probability = float(vy.mean())
            baseline_brier = float(np.mean((baseline_probability - ty) ** 2))
            # Retrieve aggregate draws in exactly the target row order.
            samples = np.vstack([
                sample_store[(scale, "test")][(horizon, str(row.aggregate_id))][date_index]
                for row in target.itertuples(index=False)
                for date_index in [
                    int(np.where(
                        np.sort(target.target_date.unique()) == np.datetime64(row.target_date)
                    )[0][0])
                ]
            ])
            observed_incidence = target.target_cases.to_numpy(float) / target.population.to_numpy(float) * 1e5
            sample_incidence = samples / target.population.to_numpy(float)[:, None] * 1e5
            prediction_incidence = samples.mean(axis=1) / target.population.to_numpy(float) * 1e5
            target["brier_loss"] = np.square(tp - ty)
            target["crps_incidence_loss"] = _crps(observed_incidence, sample_incidence)
            target["absolute_incidence_error"] = np.abs(
                prediction_incidence - observed_incidence
            )
            target["validation_prevalence"] = baseline_probability
            calibrated.append(target)
            auprc = float(average_precision_score(ty, tp))
            brier = float(np.mean((tp - ty) ** 2))
            metrics.append({
                "scale": scale, "horizon": horizon, "n": len(target),
                "spatial_units": target.aggregate_id.nunique(),
                "median_communes_per_unit": float(target.communes_per_unit.median()),
                "outbreak_prevalence": float(ty.mean()), "brier": brier,
                "validation_prevalence_baseline_brier": baseline_brier,
                "brier_skill": 1 - brier / baseline_brier if baseline_brier else np.nan,
                "auprc": auprc, "auprc_lift_over_prevalence": auprc / ty.mean(),
                "mae_incidence": float(np.mean(np.abs(prediction_incidence - observed_incidence))),
                "crps_incidence": float(_crps(observed_incidence, sample_incidence).mean()),
                "test_calibration_intercept": diag_i,
                "test_calibration_slope": diag_s, "ece_5_equal_count_bins": ece,
            })
            row = {"scale": scale, "horizon": horizon, "test_auprc": auprc}
            for specificity in (0.90, 0.95):
                cutoff, achieved = threshold_for_constraint(vy, vp, specificity=specificity)
                scored = classification(ty, tp, cutoff)
                suffix = str(int(specificity * 100))
                row[f"validation_cutoff_at_specificity_{suffix}"] = cutoff
                row[f"validation_specificity_at_cutoff_{suffix}"] = achieved["outbreak_specificity"]
                row[f"test_sensitivity_at_specificity_{suffix}"] = scored["outbreak_sensitivity"]
                row[f"test_specificity_at_specificity_{suffix}"] = scored["outbreak_specificity"]
            matched.append(row)
    return (pd.DataFrame(metrics), pd.DataFrame(matched), pd.DataFrame(calibration),
            pd.DataFrame(policies), pd.concat(calibrated, ignore_index=True))


def bootstrap_support(calibrated: pd.DataFrame, replicates: int = 2000) -> pd.DataFrame:
    """Month-block contrasts against the commune endpoint."""
    months = np.sort(calibrated.target_date.unique())
    rng = np.random.default_rng(9411)

    monthly = {}
    for scale, part in calibrated.groupby("scale"):
        work = part.copy()
        work["reference_brier_loss"] = np.square(
            work.validation_prevalence - work.observed_outbreak
        )
        monthly[scale] = work.groupby("target_date").agg(
            brier_loss=("brier_loss", "mean"),
            reference_brier_loss=("reference_brier_loss", "mean"),
            crps_incidence_loss=("crps_incidence_loss", "mean"),
            absolute_incidence_error=("absolute_incidence_error", "mean"),
        ).reindex(months)

    def score(table: pd.DataFrame, indices: np.ndarray | None = None) -> np.ndarray:
        values = table.to_numpy(float) if indices is None else table.to_numpy(float)[indices]
        skill = 1 - values[:, 0].mean() / values[:, 1].mean()
        return np.array([skill, values[:, 2].mean(), values[:, 3].mean()])

    reference = monthly["commune_554"]
    rows = []
    for scale in ("support_128", "support_32", "province_6"):
        observed = score(monthly[scale]) - score(reference)
        draws = np.empty((replicates, 3))
        for index in range(replicates):
            sampled = rng.integers(0, len(months), len(months))
            draws[index] = score(monthly[scale], sampled) - score(reference, sampled)
        for metric_index, metric in enumerate(
            ("brier_skill", "crps_incidence", "mae_incidence")
        ):
            low, high = np.quantile(draws[:, metric_index], [0.025, 0.975])
            rows.append({
                "scale_a": scale, "scale_b": "commune_554",
                "difference": "scale_a_minus_scale_b", "metric": metric,
                "estimate": observed[metric_index], "ci_lower": low, "ci_upper": high,
                "bootstrap_replicates": replicates, "block": "target calendar month",
                "interval_excludes_zero": bool(low > 0 or high < 0),
            })
    return pd.DataFrame(rows)


def trend_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    mean = metrics.groupby("scale", as_index=False).agg(
        median_communes_per_unit=("median_communes_per_unit", "mean"),
        spatial_units=("spatial_units", "mean"), brier_skill=("brier_skill", "mean"),
        auprc_lift_over_prevalence=("auprc_lift_over_prevalence", "mean"),
        crps_incidence=("crps_incidence", "mean"),
        mae_incidence=("mae_incidence", "mean"), ece=("ece_5_equal_count_bins", "mean"),
        outbreak_prevalence=("outbreak_prevalence", "mean"),
    ).sort_values("median_communes_per_unit")
    rows = []
    x = np.log2(mean.median_communes_per_unit.to_numpy(float))
    for metric in (
        "brier_skill", "auprc_lift_over_prevalence", "crps_incidence",
        "mae_incidence", "ece",
    ):
        rho, p = spearmanr(x, mean[metric].to_numpy(float))
        rows.append({"metric": metric, "spearman_rho_vs_log2_support": rho, "p_value": p})
    return mean, pd.DataFrame(rows)


def plot(mean: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.9))
    x = mean.median_communes_per_unit
    for ax, metric, label in zip(
        axes,
        ("brier_skill", "auprc_lift_over_prevalence", "crps_incidence"),
        ("Brier skill", "AUPRC / prevalence", "CRPS (incidence)"),
    ):
        ax.plot(x, mean[metric], marker="o", color="#0072B2")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Median communes per spatial unit")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    fig.suptitle("Controlled nested spatial aggregation (same predictive draws)", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--validation-samples", type=Path, required=True)
    parser.add_argument("--monthly", type=Path, required=True)
    parser.add_argument("--centroids", type=Path, required=True)
    parser.add_argument("--adjacency", type=Path, required=True)
    parser.add_argument("--settings", type=Path, help="Deprecated; retained for runner compatibility")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", default="128,32")
    args = parser.parse_args()

    test_frame = pd.read_csv(args.predictions, dtype={"fcode": str})
    validation_frame = pd.read_csv(args.validation_predictions, dtype={"fcode": str})
    test_cube = np.load(args.samples)["samples"]
    validation_cube = np.load(args.validation_samples)["samples"]
    monthly = pd.read_csv(args.monthly, dtype={"fcode": str, "l1_code": str})
    units = list(test_frame[test_frame.horizon.eq(1)].fcode.drop_duplicates())
    monthly = monthly[monthly.fcode.isin(units)].copy()
    centroids = pd.read_csv(args.centroids, dtype={"fcode": str})
    adjacency = pd.read_csv(args.adjacency, dtype=str)
    targets = tuple(int(value) for value in args.targets.split(","))
    maps = nested_maps(units, monthly, centroids, adjacency, targets)

    all_validation, all_test, sample_store = [], [], {}
    target_dates = pd.date_range("2019-01-01", "2025-12-01", freq="MS")
    hierarchy_rows = []
    for level, mapping in maps.items():
        history = aggregate_history(monthly, mapping)
        thresholds = build_primary_thresholds(history, target_dates)
        valid, valid_samples = aggregate_split(
            validation_frame, validation_cube, mapping, thresholds, level
        )
        test, test_samples = aggregate_split(test_frame, test_cube, mapping, thresholds, level)
        all_validation.append(valid); all_test.append(test)
        sample_store[(level, "validation")] = valid_samples
        sample_store[(level, "test")] = test_samples
        part = mapping.copy(); part["scale"] = level
        hierarchy_rows.append(part)
    validation = pd.concat(all_validation, ignore_index=True)
    test = pd.concat(all_test, ignore_index=True)
    metrics, matched, calibration, policies, calibrated = score_levels(
        validation, test, sample_store
    )
    bootstrap = bootstrap_support(calibrated)
    mean, trends = trend_summary(metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.concat(hierarchy_rows, ignore_index=True).to_csv(
        args.output / "nested_spatial_mapping.csv", index=False
    )
    validation.to_csv(args.output / "aggregated_validation_probabilities_raw.csv", index=False)
    test.to_csv(args.output / "aggregated_test_probabilities_raw.csv", index=False)
    calibrated.to_csv(args.output / "aggregated_test_probabilities_calibrated.csv", index=False)
    metrics.to_csv(args.output / "metrics_by_scale_horizon.csv", index=False)
    mean.to_csv(args.output / "metrics_mean_by_scale.csv", index=False)
    matched.to_csv(args.output / "matched_operating_points.csv", index=False)
    policies.to_csv(args.output / "validation_calibration_policies.csv", index=False)
    trends.to_csv(args.output / "support_trend_tests.csv", index=False)
    bootstrap.to_csv(args.output / "support_block_bootstrap.csv", index=False)
    plot(mean, args.output / "controlled_spatial_support.pdf")
    manifest = {
        "design": "nested aggregation of identical commune predictive draws; no retraining",
        "interpretation": (
            "Tests spatial-support/noise-cancellation, not whether province models learn from "
            "more independent training examples. Aggregation reduces the number of prediction units."
        ),
        "source_predictions": str(args.predictions.resolve()),
        "levels": {key: int(value.aggregate_id.nunique()) for key, value in maps.items()},
        "threshold": "Primary same-calendar-month preceding-five-year mean + 2 sample SD, rebuilt at every scale",
        "targets": list(targets), "communes": len(units),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(mean.to_string(index=False))


if __name__ == "__main__":
    main()
