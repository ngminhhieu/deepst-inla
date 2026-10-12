#!/usr/bin/env python3
"""Build the locked paper figure set from real H1--H3 forecasts and study data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from dengue_h3.evaluation import outbreak_classification_metrics


NEURAL_MODELS = ("M1", "M2", "M3", "M4", "M5", "M6")
TEMPORAL_MODELS = ("CNN", "LSTM", "CNN-LSTM")
GRAPH_BENCHMARK_MODELS = (
    "GNN", "LSTM-FULL", "LSTM-GNN", "CNN-LSTM-FULL", "CNN-LSTM-GNN"
)
INLA_MODELS = ("INLA-T", "INLA-S", "INLA-TS")
MODEL_ORDER = (
    "seasonal_naive", "province_allocation", *INLA_MODELS,
    *TEMPORAL_MODELS, *GRAPH_BENCHMARK_MODELS, *NEURAL_MODELS, "M6-INLA",
)
DISPLAY = {
    "seasonal_naive": "Seasonal\nnaive",
    "province_allocation": "Province\nallocation",
    **{model: model for model in NEURAL_MODELS},
    "CNN": "CNN",
    "LSTM": "LSTM",
    "CNN-LSTM": "CNN–LSTM",
    "GNN": "GNN",
    "LSTM-FULL": "LSTM",
    "LSTM-GNN": "LSTM–GNN",
    "CNN-LSTM-FULL": "CNN–LSTM",
    "CNN-LSTM-GNN": "CNN–LSTM–GNN",
    "INLA-T": "INLA–T",
    "INLA-S": "INLA–S",
    "INLA-TS": "INLA–TS",
    "M4": "Fixed-graph TCN",
    "M6": "Gated TCN–GNN",
    "M6-INLA": "Deep representation–INLA",
}
COLORS = {
    "seasonal_naive": "#BDBDBD",
    "province_allocation": "#8C8C8C",
    "CNN": "#44AA99",
    "LSTM": "#AA4499",
    "CNN-LSTM": "#DDCC77",
    "GNN": "#E69F00",
    "LSTM-FULL": "#AA4499",
    "LSTM-GNN": "#56B4E9",
    "CNN-LSTM-FULL": "#CC79A7",
    "CNN-LSTM-GNN": "#117733",
    "INLA-T": "#882255",
    "INLA-S": "#661100",
    "INLA-TS": "#332288",
    "M1": "#999999",
    "M2": "#009E73",
    "M3": "#CC79A7",
    "M4": "#E69F00",
    "M5": "#56B4E9",
    "M6": "#0072B2",
    "M6-INLA": "#CC3311",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument("--paper-dir", default="../paper")
    parser.add_argument("--monthly-path", required=True)
    parser.add_argument("--boundaries-path", required=True)
    parser.add_argument("--adjacency-path", required=True)
    parser.add_argument("--names-path", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    results = Path(args.results_dir).resolve()
    paper = Path(args.paper_dir).resolve()
    figures = paper / "figures"
    tables = paper / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    predictions = _load_predictions(results)
    all_metrics = _load_all_metrics(results)
    metrics = all_metrics[all_metrics["horizon"] == 3].copy()
    peak = _peak_fidelity(predictions)
    all_metrics.to_csv(tables / "Table2_all_horizons.csv", index=False)
    metrics.to_csv(tables / "Table2_primary_h3.csv", index=False)
    peak.to_csv(tables / "Table3_h3_peak_fidelity.csv", index=False)

    monthly = pd.read_csv(args.monthly_path, dtype={"fcode": str})
    monthly["date"] = pd.to_datetime(
        dict(year=monthly["year"], month=monthly["month"], day=1)
    )
    boundaries = _load_boundaries(Path(args.boundaries_path))

    _epidemic_context_figure(
        monthly, boundaries, figures / "Fig1_spatiotemporal_motivation"
    )
    _motivation_figure(
        monthly, boundaries, Path(args.adjacency_path), Path(args.names_path),
        figures / "Fig2_spatial_motivation",
    )
    _comparison_figure(
        metrics, predictions, figures / "Fig3_h3_model_comparison"
    )
    component_table = tables / "TableS1_h3_component_changes.csv"
    component = _component_figure(
        results,
        predictions,
        metrics,
        figures / "Fig4_h3_ablation",
        args.bootstrap_draws,
        args.seed,
        component_table,
    )
    component.to_csv(component_table, index=False)
    _spatial_figure(predictions, boundaries, figures / "Fig5_h3_spatial_performance")
    _calibration_figure(predictions, figures / "Fig7_h3_calibration")
    _encoder_benchmark_figure(
        results, metrics, predictions, figures / "Fig8_temporal_benchmark"
    )
    print(f"wrote locked paper figures to {figures}")


def _load_predictions(results: Path) -> dict[str, pd.DataFrame]:
    output = {}
    for model in MODEL_ORDER:
        path = results / model / "predictions.csv"
        if not path.exists():
            raise SystemExit(f"Required result is missing: {path}")
        frame = pd.read_csv(path, dtype={"fcode": str})
        frame["target_date"] = pd.to_datetime(frame["target_date"])
        output[model] = frame
    return output


def _load_all_metrics(results: Path) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        path = results / model / "metrics.csv"
        if not path.exists():
            raise SystemExit(f"Required result is missing: {path}")
        frame = pd.read_csv(path)
        expected = {1, 2, 3}
        found = set(frame["horizon"].astype(int))
        if found != expected:
            raise ValueError(f"Expected H1--H3 metric rows for {model}; found {sorted(found)}")
        rows.extend(frame.to_dict("records"))
    output = pd.DataFrame(rows).reset_index(drop=True)
    output["model"] = pd.Categorical(output["model"], MODEL_ORDER, ordered=True)
    return output.sort_values(["model", "horizon"]).reset_index(drop=True)


def _peak_fidelity(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model in NEURAL_MODELS:
        frame = predictions[model]
        frame = frame[frame["horizon"] == 3].copy()
        frame["observed_incidence"] = frame["target_cases"] / frame["population"] * 1e5
        frame["predicted_incidence"] = (
            frame["predicted_mean"] / frame["population"] * 1e5
        )
        unit_rows = []
        for unit, part in frame.groupby("fcode"):
            part = part.sort_values("target_date").reset_index(drop=True)
            observed = part["observed_incidence"].to_numpy(float)
            predicted = part["predicted_incidence"].to_numpy(float)
            observed_peak = int(np.argmax(observed))
            predicted_peak = int(np.argmax(predicted))
            timing = abs(
                (
                    part.loc[predicted_peak, "target_date"].to_period("M")
                    - part.loc[observed_peak, "target_date"].to_period("M")
                ).n
            )
            ratio = predicted[predicted_peak] / observed[observed_peak]
            correlation = (
                np.corrcoef(observed, predicted)[0, 1]
                if observed.std() > 0 and predicted.std() > 0
                else np.nan
            )
            unit_rows.append((unit, timing, ratio, correlation))
        unit = pd.DataFrame(
            unit_rows,
            columns=("fcode", "peak_timing_error", "peak_amplitude_ratio", "correlation"),
        )
        rows.append(
            {
                "model": model,
                "communes": len(unit),
                "median_peak_timing_error_months": unit["peak_timing_error"].median(),
                "median_peak_amplitude_ratio": unit["peak_amplitude_ratio"].median(),
                "median_temporal_correlation": unit["correlation"].median(),
                "peak_ratio_q25": unit["peak_amplitude_ratio"].quantile(0.25),
                "peak_ratio_q75": unit["peak_amplitude_ratio"].quantile(0.75),
            }
        )
    return pd.DataFrame(rows)


def _load_boundaries(path: Path) -> dict[str, list[np.ndarray]]:
    with path.open(encoding="utf-8") as handle:
        geojson = json.load(handle)
    output: dict[str, list[np.ndarray]] = {}
    for feature in geojson["features"]:
        unit = str(feature["properties"]["fcode"])
        geometry = feature["geometry"]
        coordinates = geometry["coordinates"]
        if geometry["type"] == "Polygon":
            polygons = [coordinates]
        elif geometry["type"] == "MultiPolygon":
            polygons = coordinates
        else:
            continue
        output[unit] = [np.asarray(polygon[0], dtype=float) for polygon in polygons]
    return output


def _epidemic_context_figure(monthly, boundaries, output):
    monthly = monthly[monthly["date"] <= pd.Timestamp("2025-12-01")].copy()
    monthly["incidence"] = monthly["obs_dengue_cases"] / monthly["population"] * 1e5
    centroids = _boundary_centroids(boundaries)
    units = [unit for unit in centroids if unit in set(monthly["fcode"])]
    units = sorted(units, key=lambda unit: (centroids[unit][0], centroids[unit][1]))
    dates = np.sort(monthly["date"].unique())
    carpet = (
        monthly.pivot(index="fcode", columns="date", values="incidence")
        .reindex(index=units, columns=dates)
        .to_numpy(float)
    )
    carpet_log = np.log1p(carpet)
    carpet_limit = np.nanquantile(carpet_log, .98)

    test = monthly[monthly["date"].between("2023-01-01", "2025-12-01")]
    peak_rows = test.loc[test.groupby("fcode")["incidence"].idxmax()].copy()
    peak_rows["peak_month"] = (
        (peak_rows["date"].dt.year - 2023) * 12 + peak_rows["date"].dt.month - 1
    )
    peak_rows = peak_rows.set_index("fcode")

    fig, axes = plt.subplots(
        1, 2, figsize=(15, 7.2), gridspec_kw={"width_ratios": (1.35, 1)},
        constrained_layout=True,
    )
    date_numbers = mdates.date2num(pd.to_datetime(dates))
    image = axes[0].imshow(
        carpet_log, aspect="auto", origin="lower",
        extent=(date_numbers[0], date_numbers[-1], 0, len(units)),
        cmap="magma", vmin=0, vmax=carpet_limit, interpolation="nearest",
    )
    axes[0].axvline(
        mdates.date2num(pd.Timestamp("2023-01-01")),
        color="#00E5FF", linewidth=1.5,
    )
    axes[0].text(
        mdates.date2num(pd.Timestamp("2023-02-01")), len(units) * .97,
        "Held-out test", color="#00E5FF", va="top", fontweight="bold",
    )
    axes[0].xaxis_date()
    axes[0].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].set(
        xlabel="Target month", ylabel="Communes ordered west → east",
        yticks=(0, len(units) - 1), yticklabels=("West", "East"),
        title="A. Space–time incidence fingerprint",
    )
    carpet_bar = fig.colorbar(image, ax=axes[0], shrink=.75, pad=.02)
    carpet_ticks = np.linspace(0, carpet_limit, 5)
    carpet_bar.set_ticks(carpet_ticks)
    carpet_bar.set_ticklabels([f"{np.expm1(value):.0f}" for value in carpet_ticks])
    carpet_bar.set_label("Monthly incidence / 100,000")

    # Map the observed peak month directly to each commune polygon.  We avoid
    # interpolating between commune centroids because interpolation can imply a
    # continuous transmission wave that is not identified by these data.
    map_polygons = []
    map_values = []
    for unit in units:
        if unit not in peak_rows.index:
            continue
        value = float(peak_rows.loc[unit, "peak_month"])
        for polygon in boundaries.get(unit, []):
            map_polygons.append(polygon)
            map_values.append(value)
    peak_norm = Normalize(vmin=0, vmax=35)
    peak_map = PolyCollection(
        map_polygons,
        array=np.asarray(map_values, dtype=float),
        cmap="viridis",
        norm=peak_norm,
        edgecolors="#FFFFFF",
        linewidths=.20,
    )
    axes[1].add_collection(peak_map)
    axes[1].autoscale_view()

    plotted_units = [unit for unit in units if unit in peak_rows.index]
    x = np.array([centroids[unit][0] for unit in plotted_units])
    y = np.array([centroids[unit][1] for unit in plotted_units])
    peak_incidence = peak_rows.loc[plotted_units, "incidence"].to_numpy(float)
    clipped_peak = np.clip(
        peak_incidence, 0, np.nanquantile(peak_incidence, .98)
    )
    # Matplotlib's ``s`` is marker area, so scale it linearly with incidence.
    if np.nanmax(clipped_peak) > 0:
        size_scale = 7 + 42 * clipped_peak / np.nanmax(clipped_peak)
    else:
        size_scale = np.full_like(size_scale, 7.0)
    axes[1].scatter(
        x, y, s=size_scale, facecolors="none", edgecolors="#111111",
        linewidths=.35, alpha=.55, zorder=3,
    )
    axes[1].set_aspect("equal")
    axes[1].set_axis_off()
    axes[1].set_title("B. Observed peak month by commune (2023–2025)")
    peak_bar = fig.colorbar(peak_map, ax=axes[1], shrink=.75, pad=.02)
    peak_ticks = np.append(np.arange(0, 36, 6), 35)
    peak_bar.set_ticks(peak_ticks)
    peak_bar.set_ticklabels([
        (pd.Timestamp("2023-01-01") + pd.DateOffset(months=int(value))).strftime("%b %Y")
        for value in peak_ticks
    ])
    peak_bar.set_label("Month of maximum observed incidence")
    axes[1].text(
        .02, .02,
        "Polygon colour = observed peak month\nCircle size ∝ peak incidence\nNo spatial interpolation",
        transform=axes[1].transAxes, fontsize=8, va="bottom",
        bbox={"facecolor": "white", "alpha": .78, "edgecolor": "none"},
    )
    fig.suptitle("Local dengue waves differ in timing, magnitude, and spatial footprint")
    _save(fig, output)


def _motivation_figure(monthly, boundaries, adjacency_path, names_path, output):
    monthly = monthly[monthly["date"] <= pd.Timestamp("2025-12-01")].copy()
    monthly["incidence"] = monthly["obs_dengue_cases"] / monthly["population"] * 1e5
    commune = monthly.groupby("fcode").agg(
        mean_population=("population", "mean"),
        cumulative_cases=("obs_dengue_cases", "sum"),
        mean_incidence=("incidence", "mean"),
    )
    log_population = np.log10(commune["mean_population"])
    log_cases = np.log10(1 + commune["cumulative_cases"])
    slope, intercept = np.polyfit(log_population, log_cases, 1)
    rank_correlation = commune[["mean_population", "cumulative_cases"]].corr(
        method="spearman"
    ).iloc[0, 1]

    centroids = _boundary_centroids(boundaries)
    names = pd.read_csv(names_path, dtype={"fcode": str}).set_index("fcode")
    adjacency = pd.read_csv(adjacency_path, dtype=str)
    left, right = ("from", "to") if {"from", "to"} <= set(adjacency) else (
        "from_fcode", "to_fcode"
    )
    pairs = np.sort(adjacency[[left, right]].astype(str).to_numpy(), axis=1)
    pairs = pd.DataFrame(pairs, columns=("left", "right")).drop_duplicates()
    incidence = monthly.pivot(index="date", columns="fcode", values="incidence")
    edges = []
    for left_code, right_code in pairs.itertuples(index=False):
        if left_code not in incidence or right_code not in incidence:
            continue
        if left_code not in centroids or right_code not in centroids:
            continue
        edge_correlation = np.corrcoef(
            np.log1p(incidence[left_code]), np.log1p(incidence[right_code])
        )[0, 1]
        distance = _haversine_pair(centroids[left_code], centroids[right_code])
        edges.append((left_code, right_code, distance, edge_correlation))
    edges = pd.DataFrame(edges, columns=("left", "right", "distance", "correlation"))
    weak = edges.loc[edges["correlation"].idxmin()]
    long_weak = edges[edges["correlation"] < edges["correlation"].quantile(.25)].nlargest(
        1, "distance"
    ).iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    axes[0].hexbin(
        log_population, log_cases, gridsize=30, mincnt=1, bins="log", cmap="magma"
    )
    population_grid = np.linspace(log_population.min(), log_population.max(), 100)
    axes[0].plot(
        population_grid, intercept + slope * population_grid,
        color="#00A6D6", linewidth=2.2,
    )
    axes[0].set(
        xlabel="log10(mean commune population)",
        ylabel="log10(1 + cumulative dengue cases)",
        title="A. Population explains only part of dengue burden",
    )
    axes[0].text(
        .04, .96, f"Spearman ρ = {rank_correlation:.2f}\nfit slope = {slope:.2f}",
        transform=axes[0].transAxes, va="top",
        bbox={"facecolor": "white", "alpha": .86, "edgecolor": "none"},
    )

    density = axes[1].hexbin(
        edges["distance"], edges["correlation"], gridsize=32, mincnt=1,
        bins="log", cmap="Blues",
    )
    axes[1].axhline(0, color="#555555", linewidth=.8)
    for row, color, label in (
        (weak, "#CC3311", "weakest temporal match"),
        (long_weak, "#E69F00", "long, weak edge"),
    ):
        axes[1].scatter(row.distance, row.correlation, s=65, color=color,
                        edgecolor="white", linewidth=.8, zorder=4)
        axes[1].annotate(
            label, (row.distance, row.correlation), xytext=(7, 7),
            textcoords="offset points", fontsize=8, color=color,
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": .6},
        )
    axes[1].set(
        xlabel="Centroid distance along an administrative edge (km)",
        ylabel="Correlation of monthly log-incidence",
        title="B. Adjacency does not imply epidemic similarity",
    )
    edge_bar = fig.colorbar(density, ax=axes[1], shrink=.76, pad=.02)
    edge_bar.set_label("Administrative-edge density")

    weak_codes = (str(weak.left), str(weak.right))
    split_colours = (("2010-01-01", "2020-12-01", "#E5E5E5", "Train"),
                     ("2021-01-01", "2022-12-01", "#FFF2CC", "Validation"),
                     ("2023-01-01", "2025-12-01", "#DDEBF7", "Test"))
    for start, end, colour, label in split_colours:
        axes[2].axvspan(pd.Timestamp(start), pd.Timestamp(end), color=colour,
                        alpha=.65, label=label)
    for code, colour in zip(weak_codes, ("#0072B2", "#D55E00")):
        series = monthly[monthly["fcode"] == code].sort_values("date")
        display_name = names["region_name"].get(code, code)
        axes[2].plot(series["date"], np.log1p(series["incidence"]),
                     color=colour, linewidth=1.5, label=display_name)
    axes[2].xaxis.set_major_locator(mdates.YearLocator(3))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[2].set(
        xlabel="Month and experimental split",
        ylabel="log(1 + monthly incidence / 100,000)",
        title=f"C. Adjacent communes can follow different trajectories (r={weak.correlation:.2f})",
    )
    handles, labels = axes[2].get_legend_handles_labels()
    axes[2].legend(handles[-5:], labels[-5:], frameon=False, fontsize=7, loc="upper left")
    for axis in axes:
        axis.grid(alpha=.16)
    fig.suptitle(
        "Why fixed linear spatial smoothing is insufficient for commune dengue forecasts"
    )
    _save(fig, output)


def _comparison_figure(metrics, predictions, output):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=True)

    maxima = []
    for axis, model, panel in zip(axes[:2], ("INLA-TS", "M6"), ("A", "B")):
        maxima.append(_peak_centered_profile(axis, predictions[model], model, panel))
    common_ymax = max(1.15, *(value for value in maxima if np.isfinite(value)))
    for axis in axes[:2]:
        axis.set_ylim(0, common_ymax)

    axis = axes[2]
    decision_models = (
        "seasonal_naive", "province_allocation", "INLA-T", "INLA-S",
        "INLA-TS", "M4", "M6", "M6-INLA",
    )
    h3 = metrics[metrics["model"].astype(str).isin(decision_models)].copy()
    for f1_level in (.2, .3, .4):
        sensitivity_grid = np.linspace(f1_level / 2 + .002, .36, 200)
        precision_grid = f1_level * sensitivity_grid / (2 * sensitivity_grid - f1_level)
        valid = (precision_grid >= 0) & (precision_grid <= .62)
        axis.plot(
            sensitivity_grid[valid], precision_grid[valid], color="#B0B0B0",
            linewidth=.8, linestyle=":",
        )
        if valid.any():
            axis.text(
                sensitivity_grid[valid][-1], precision_grid[valid][-1],
                f" F1={f1_level:.1f}", fontsize=7, color="#777777",
            )
    label_offsets = {
        "seasonal_naive": (5, 4),
        "province_allocation": (6, 5),
        "M4": (-72, 7),
        "M6": (9, -19),
        "M6-INLA": (9, 8),
        "INLA-T": (-28, -17),
        "INLA-S": (-20, -17),
        "INLA-TS": (7, -15),
    }
    for row in h3.itertuples(index=False):
        model = str(row.model)
        f1 = float(row.outbreak_f1)
        axis.scatter(
            row.outbreak_sensitivity, row.outbreak_precision,
            s=80 + 850 * f1, color=COLORS[model], edgecolor="white",
            linewidth=1, alpha=.92, zorder=3,
        )
        axis.annotate(
            DISPLAY[model].replace("\n", " "),
            (row.outbreak_sensitivity, row.outbreak_precision),
            xytext=label_offsets[model], textcoords="offset points",
            fontsize=7,
            arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": .45},
        )
    axis.set(
        xlabel="Sensitivity: fraction of observed outbreaks detected",
        ylabel="Precision: fraction of alerts that were outbreaks",
        xlim=(-.015, .36), ylim=(-.02, .75),
        title="C. Alert trade-off at probability threshold 0.50",
    )
    axis.grid(alpha=.18)
    axis.text(
        .02, .98, "Bubble area ∝ F1", transform=axis.transAxes,
        va="top", fontsize=8,
    )
    fig.suptitle("H3 outbreak-shape fidelity and the operational precision–recall trade-off")
    _save(fig, output)


def _peak_centered_profile(axis, frame, model, panel, window=4):
    """Show typical model behaviour around each commune's observed H3 peak.

    Each commune is aligned on its maximum observed incidence in the held-out
    H3 period.  Observed and predicted incidence are divided by that commune's
    observed peak, so the panel describes peak *shape and shrinkage* rather
    than being dominated by a few high-incidence communes.
    """
    frame = frame[frame["horizon"] == 3].copy()
    frame["observed_incidence"] = (
        frame["target_cases"].to_numpy(float)
        / frame["population"].to_numpy(float) * 1e5
    )
    frame["predicted_incidence"] = (
        frame["predicted_mean"].to_numpy(float)
        / frame["population"].to_numpy(float) * 1e5
    )
    frame = frame.sort_values(["fcode", "target_date"]).reset_index(drop=True)

    profiles = []
    peak_capture = []
    timing_error = []
    for unit, part in frame.groupby("fcode", sort=False):
        part = part.sort_values("target_date").reset_index(drop=True)
        observed = part["observed_incidence"].to_numpy(float)
        predicted = part["predicted_incidence"].to_numpy(float)
        if not len(part) or not np.isfinite(observed).any():
            continue
        observed_peak_index = int(np.nanargmax(observed))
        observed_peak = float(observed[observed_peak_index])
        if not np.isfinite(observed_peak) or observed_peak <= 0:
            continue

        predicted_peak_index = int(np.nanargmax(predicted))
        observed_peak_date = part.loc[observed_peak_index, "target_date"]
        predicted_peak_date = part.loc[predicted_peak_index, "target_date"]
        timing_error.append(abs((predicted_peak_date.to_period("M") - observed_peak_date.to_period("M")).n))
        peak_capture.append(float(predicted[observed_peak_index] / observed_peak))

        peak_month_index = observed_peak_date.year * 12 + observed_peak_date.month
        month_index = part["target_date"].dt.year * 12 + part["target_date"].dt.month
        offsets = (month_index - peak_month_index).to_numpy(int)
        selected = np.abs(offsets) <= window
        for offset, obs_value, pred_value in zip(
            offsets[selected], observed[selected], predicted[selected]
        ):
            profiles.append({
                "offset": int(offset),
                "observed": float(obs_value / observed_peak),
                "predicted": float(pred_value / observed_peak),
            })

    profile = pd.DataFrame(profiles)
    if profile.empty:
        axis.text(.5, .5, "No valid peak profiles", ha="center", va="center")
        return 1.15

    summary = profile.groupby("offset").agg(
        observed_median=("observed", "median"),
        observed_q25=("observed", lambda x: np.quantile(x, .25)),
        observed_q75=("observed", lambda x: np.quantile(x, .75)),
        predicted_median=("predicted", "median"),
        predicted_q25=("predicted", lambda x: np.quantile(x, .25)),
        predicted_q75=("predicted", lambda x: np.quantile(x, .75)),
    ).reindex(range(-window, window + 1))
    offsets = summary.index.to_numpy(float)

    axis.fill_between(
        offsets,
        summary["observed_q25"].to_numpy(float),
        summary["observed_q75"].to_numpy(float),
        color="#777777", alpha=.12, linewidth=0,
    )
    axis.fill_between(
        offsets,
        summary["predicted_q25"].to_numpy(float),
        summary["predicted_q75"].to_numpy(float),
        color=COLORS[model], alpha=.16, linewidth=0,
    )
    axis.plot(
        offsets, summary["observed_median"], color="#222222",
        linewidth=2.0, marker="o", markersize=3.5, label="Observed",
    )
    axis.plot(
        offsets, summary["predicted_median"], color=COLORS[model],
        linewidth=2.2, marker="o", markersize=3.5,
        label=DISPLAY[model].replace("\n", " "),
    )
    axis.axvline(0, color="#777777", linewidth=.8, linestyle="--")
    axis.axhline(1, color="#B0B0B0", linewidth=.7, linestyle=":")
    axis.set(
        xlabel="Months from each commune's observed H3 peak",
        ylabel="Incidence / observed commune peak" if panel == "A" else "",
        xticks=np.arange(-window, window + 1, 2),
        title=f"{panel}. {DISPLAY[model].replace(chr(10), ' ')} around observed peaks",
    )
    axis.grid(alpha=.14)
    axis.legend(frameon=False, loc="upper left", fontsize=8)

    capture = np.asarray(peak_capture, dtype=float)
    timing = np.asarray(timing_error, dtype=float)
    capture = capture[np.isfinite(capture)]
    timing = timing[np.isfinite(timing)]
    capture_text = f"{np.median(capture):.2f}×" if len(capture) else "NA"
    timing_text = f"{np.median(timing):.1f} mo" if len(timing) else "NA"
    axis.text(
        .98, .96,
        f"Median prediction at observed peak = {capture_text}\n"
        f"Median peak-timing error = {timing_text}",
        transform=axis.transAxes, ha="right", va="top", fontsize=8,
        bbox={"facecolor": "white", "alpha": .84, "edgecolor": "none"},
    )

    ymax = np.nanmax([
        summary["observed_q75"].max(), summary["predicted_q75"].max(), 1.0
    ])
    return float(max(1.15, ymax * 1.12))


def _component_figure(results, predictions, metrics, output, draws, seed, cached_table):
    sample_paths = [results / model / "predictive_samples.npz" for model in NEURAL_MODELS]
    if all(path.exists() for path in sample_paths):
        row_scores = {
            model: _row_crps(results / model, predictions[model])
            for model in NEURAL_MODELS
        }
        rng = np.random.default_rng(seed)
        rows = []
        for previous, current in zip(NEURAL_MODELS[:-1], NEURAL_MODELS[1:]):
            merged = row_scores[previous].merge(
                row_scores[current], on=("fcode", "target_date", "year"),
                suffixes=("_previous", "_current"), validate="one_to_one"
            )
            merged["difference"] = merged["crps_row_current"] - merged["crps_row_previous"]
            samples = _two_way_bootstrap(merged, draws, rng)
            rows.append({
                "comparison": f"{previous}→{current}",
                "mean_h3_crps_change": merged["difference"].mean(),
                "lower_95": np.quantile(samples, .025),
                "upper_95": np.quantile(samples, .975),
            })
        changes = pd.DataFrame(rows)
    elif Path(cached_table).exists():
        changes = pd.read_csv(cached_table)
    else:
        raise SystemExit(
            "Component bootstrap requires predictive samples or the locked "
            f"derived table {cached_table}"
        )
    peak = _peak_fidelity(predictions)
    neural = metrics[metrics["model"].astype(str).isin(NEURAL_MODELS)].copy()
    neural["model"] = neural["model"].astype(str)
    neural = neural.merge(peak, on="model").set_index("model").loc[list(NEURAL_MODELS)]
    columns = [
        ("CRPS↓", "crps", "low", ".3f"),
        ("MAE↓", "mae", "low", ".2f"),
        ("F1↑", "outbreak_f1", "high", ".3f"),
        ("Precision↑", "outbreak_precision", "high", ".3f"),
        ("Specificity↑", "outbreak_specificity", "high", ".3f"),
        ("Recall↑", "outbreak_recall", "high", ".3f"),
        ("Peak corr.↑", "median_temporal_correlation", "high", ".3f"),
        ("Peak ratio→1", "median_peak_amplitude_ratio", "target", ".3f"),
    ]
    raw = np.column_stack([neural[name].to_numpy(float) for _, name, _, _ in columns])
    desirability = np.empty_like(raw)
    for index, (_, _, direction, _) in enumerate(columns):
        values = raw[:, index]
        if direction == "low":
            score = values.max() - values
        elif direction == "high":
            score = values - values.min()
        else:
            distance = np.abs(values - 1)
            score = distance.max() - distance
        span = score.max() - score.min()
        desirability[:, index] = score / span if span else .5

    fig, axis = plt.subplots(figsize=(12.8, 5.5), constrained_layout=True)
    image = axis.imshow(desirability, cmap="RdYlBu", vmin=0, vmax=1, aspect="auto")
    components = (
        "M1  seasonal memory", "M2  + climate", "M3  + commune conditioning",
        "M4  + fixed graph", "M5  + learned graph", "M6  + dynamic gate",
    )
    axis.set_xticks(np.arange(len(columns)), [item[0] for item in columns])
    axis.set_yticks(np.arange(len(NEURAL_MODELS)), components)
    axis.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    for row in range(raw.shape[0]):
        for column, (_, _, _, fmt) in enumerate(columns):
            value = raw[row, column]
            color = "white" if desirability[row, column] < .25 else "#111111"
            axis.text(column, row, format(value, fmt), ha="center", va="center",
                      color=color, fontsize=9, fontweight="bold")
    best_rows = desirability.argmax(axis=0)
    for column, row in enumerate(best_rows):
        axis.add_patch(Rectangle((column - .48, row - .48), .96, .96,
                                 fill=False, edgecolor="#00E5FF", linewidth=2.3))
    bar = fig.colorbar(image, ax=axis, shrink=.78, pad=.02)
    bar.set_label("Relative desirability within each metric")
    axis.set_title("H3 ablation scorecard: each added component changes a different trade-off", pad=20)
    axis.text(.995, -.08, "Cyan outline = best observed value in column",
              transform=axis.transAxes, ha="right", fontsize=8)
    _save(fig, output)
    return changes


def _row_crps(model_dir, predictions):
    samples = np.load(model_dir / "predictive_samples.npz")["samples"]
    h3_index = sorted(predictions["horizon"].unique()).index(3)
    frame = predictions[predictions["horizon"] == 3].reset_index(drop=True).copy()
    values = samples[:, :, h3_index].reshape(-1, samples.shape[-1])
    population = frame["population"].to_numpy(float)
    observed = frame["target_cases"].to_numpy(float) / population * 1e5
    values = values / population[:, None] * 1e5
    ordered = np.sort(values, axis=1)
    count = ordered.shape[1]
    weights = 2 * np.arange(1, count + 1) - count - 1
    frame["crps_row"] = (
        np.abs(values - observed[:, None]).mean(axis=1)
        - (ordered * weights).sum(axis=1) / count**2
    )
    frame["year"] = frame["target_date"].dt.year
    return frame[["fcode", "target_date", "year", "crps_row"]]


def _two_way_bootstrap(frame, draws, rng):
    communes = frame["fcode"].unique()
    years = frame["year"].unique()
    commune_index = pd.Categorical(frame["fcode"], communes).codes
    year_index = pd.Categorical(frame["year"], years).codes
    values = frame["difference"].to_numpy(float)
    output = np.empty(draws)
    for draw in range(draws):
        weights = rng.poisson(1, len(communes))[commune_index] * rng.poisson(1, len(years))[year_index]
        output[draw] = np.average(values, weights=weights) if weights.sum() else values.mean()
    return output


def _spatial_figure(predictions, boundaries, output):
    summary = {}
    for model in ("M4", "M6"):
        frame = predictions[model]
        frame = frame[frame["horizon"] == 3].copy()
        frame["ae"] = abs(frame["predicted_mean"] - frame["target_cases"]) / frame["population"] * 1e5
        frame["outbreak"] = (frame["target_cases"] > frame["outbreak_threshold"]).astype(float)
        frame["brier"] = (frame["outbreak_probability"] - frame["outbreak"]) ** 2
        summary[model] = frame.groupby("fcode").agg(mae=("ae", "mean"), brier=("brier", "mean"))
    delta = summary["M6"]["mae"] - summary["M4"]["mae"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.3), constrained_layout=True)
    mae_norm = Normalize(0, summary["M4"]["mae"].quantile(.98))
    brier_norm = Normalize(0, summary["M4"]["brier"].quantile(.98))
    limit = np.quantile(abs(delta), .95)
    delta_norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    collections = [
        _draw_map(axes[0], boundaries, summary["M4"]["mae"].to_dict(), "viridis", mae_norm),
        _draw_map(axes[1], boundaries, summary["M4"]["brier"].to_dict(), "magma", brier_norm),
        _draw_map(axes[2], boundaries, delta.to_dict(), "RdBu_r", delta_norm),
    ]
    axes[0].set_title("M4 H3 MAE")
    axes[1].set_title("M4 H3 outbreak Brier score")
    axes[2].set_title("M6 − M4 H3 MAE")
    for axis, collection, label in zip(
        axes, collections,
        ("Incidence / 100,000", "Brier score", "Positive values favour M4"),
    ):
        bar = fig.colorbar(collection, ax=axis, shrink=.66, pad=.01)
        bar.set_label(label)
    fig.suptitle("Spatial heterogeneity in held-out H3 forecast performance")
    _save(fig, output)


def _draw_map(axis, boundaries, values, cmap, norm):
    polygons, colors = [], []
    for unit, parts in boundaries.items():
        value = values.get(unit, np.nan)
        for polygon in parts:
            polygons.append(polygon)
            colors.append(value)
    collection = PolyCollection(
        polygons, array=np.asarray(colors), cmap=cmap, norm=norm,
        edgecolor="#FFFFFF", linewidth=.12
    )
    collection.set_clim(norm.vmin, norm.vmax)
    axis.add_collection(collection)
    axis.autoscale_view()
    axis.set_aspect("equal")
    axis.set_axis_off()
    return collection


def _boundary_centroids(boundaries):
    output = {}
    for unit, parts in boundaries.items():
        coordinates = np.concatenate(parts, axis=0)
        output[unit] = (float(coordinates[:, 0].mean()), float(coordinates[:, 1].mean()))
    return output


def _haversine_pair(left, right):
    left_lon, left_lat = np.radians(left)
    right_lon, right_lat = np.radians(right)
    latitude = right_lat - left_lat
    longitude = right_lon - left_lon
    value = (
        np.sin(latitude / 2) ** 2
        + np.cos(left_lat) * np.cos(right_lat) * np.sin(longitude / 2) ** 2
    )
    return float(6371 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1))))


def _draw_outline(axis, boundaries):
    polygons = [polygon for parts in boundaries.values() for polygon in parts]
    outline = PolyCollection(
        polygons, facecolors="none", edgecolors="#333333", linewidths=.16, alpha=.65
    )
    axis.add_collection(outline)
    axis.autoscale_view()
    return outline


def _calibration_figure(predictions, output):
    models = (*NEURAL_MODELS, "M6-INLA")
    probability_bins = np.linspace(0, 1, 11)
    decision_thresholds = np.arange(.05, 1.0, .05)
    residual = np.full((len(models), 10), np.nan)
    bin_counts = np.zeros((len(models), 10), dtype=int)
    f1_surface = np.zeros((len(models), len(decision_thresholds)))

    for row, model in enumerate(models):
        frame = predictions[model]
        frame = frame[frame["horizon"] == 3].copy()
        observed = (frame["target_cases"] > frame["outbreak_threshold"]).to_numpy(float)
        probability = frame["outbreak_probability"].to_numpy(float)
        bin_index = np.clip(np.digitize(probability, probability_bins) - 1, 0, 9)
        for column in range(10):
            selected = bin_index == column
            if selected.any():
                residual[row, column] = observed[selected].mean() - probability[selected].mean()
                bin_counts[row, column] = selected.sum()
        for column, threshold in enumerate(decision_thresholds):
            result = outbreak_classification_metrics(
                frame["target_cases"].to_numpy(float),
                frame["outbreak_threshold"].to_numpy(float),
                probability,
                decision_threshold=float(threshold),
            )
            f1_surface[row, column] = result["outbreak_f1"]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), constrained_layout=True)
    reliability = axes[0].imshow(
        residual, aspect="auto", cmap="RdBu_r", vmin=-.25, vmax=.25
    )
    axes[0].set_xticks(np.arange(10), [f"{left:.1f}–{left + .1:.1f}" for left in probability_bins[:-1]], rotation=40, ha="right")
    axes[0].set_yticks(np.arange(len(models)), models)
    axes[0].set(xlabel="Predicted outbreak-probability bin",
                title="A. Reliability residual: observed frequency − prediction")
    for row in range(len(models)):
        for column in range(10):
            if bin_counts[row, column]:
                value = residual[row, column]
                color = "white" if abs(value) > .14 else "#111111"
                axes[0].text(column, row, f"{value:+.2f}", ha="center", va="center",
                             fontsize=7, color=color)
    residual_bar = fig.colorbar(reliability, ax=axes[0], shrink=.78, pad=.02)
    residual_bar.set_label("Overprediction ← 0 → underprediction")

    landscape = axes[1].imshow(
        f1_surface, aspect="auto", cmap="viridis", vmin=0,
        vmax=max(.45, np.nanmax(f1_surface))
    )
    axes[1].set_xticks(
        np.arange(1, len(decision_thresholds), 2),
        [f"{value:.2f}" for value in decision_thresholds[1::2]],
    )
    axes[1].set_yticks(np.arange(len(models)), models)
    threshold_index = int(np.flatnonzero(np.isclose(decision_thresholds, .5))[0])
    axes[1].axvline(threshold_index, color="white", linewidth=2)
    axes[1].set(xlabel="Probability threshold used to issue an alert",
                title="B. Exploratory F1 landscape (test set)")
    for row in range(len(models)):
        axes[1].text(threshold_index, row, f"{f1_surface[row, threshold_index]:.2f}",
                     ha="center", va="center", fontsize=8, fontweight="bold",
                     color="white" if f1_surface[row, threshold_index] < .32 else "#111111")
    axes[1].text(threshold_index + .2, -.65, "fixed 0.50", color="white", fontsize=8)
    f1_bar = fig.colorbar(landscape, ax=axes[1], shrink=.78, pad=.02)
    f1_bar.set_label("F1 score")
    fig.suptitle("H3 outbreak forecasts: probability calibration and alert-threshold sensitivity")
    _save(fig, output)


def _encoder_benchmark_figure(results, metrics, predictions, output):
    pairs = (
        ("LSTM-FULL", "LSTM-GNN", "LSTM", "#AA4499"),
        ("CNN-LSTM-FULL", "CNN-LSTM-GNN", "CNN–LSTM", "#117733"),
        ("M3", "M6", "TCN", "#0072B2"),
    )
    required = {"GNN", *(item for pair in pairs for item in pair[:2])}
    frames = []
    for model in sorted(required):
        frame = pd.read_csv(results / model / "metrics.csv")
        frame["model"] = model
        frames.append(frame)
    scores = pd.concat(frames, ignore_index=True).set_index(["model", "horizon"])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    for horizon, axis in zip((1, 2, 3), axes):
        horizon_rows = scores.xs(horizon, level="horizon")
        axis.axvspan(
            horizon_rows["crps"].min() - .08, horizon_rows["crps"].median(),
            color="#E8F4F8", alpha=.45, zorder=0,
        )
        axis.axhspan(
            horizon_rows["outbreak_f1"].median(),
            horizon_rows["outbreak_f1"].max() + .025,
            color="#E8F4F8", alpha=.45, zorder=0,
        )
        for base, graph, family, colour in pairs:
            start = horizon_rows.loc[base]
            end = horizon_rows.loc[graph]
            axis.annotate(
                "", xy=(end.crps, end.outbreak_f1),
                xytext=(start.crps, start.outbreak_f1),
                arrowprops={"arrowstyle": "->", "color": colour, "linewidth": 1.6},
                zorder=2,
            )
            axis.scatter(
                start.crps, start.outbreak_f1, s=75, facecolor="white",
                edgecolor=colour, linewidth=1.7, zorder=3,
            )
            axis.scatter(
                end.crps, end.outbreak_f1, s=95, color=colour,
                edgecolor="white", linewidth=.8, zorder=4,
            )
            axis.annotate(
                family, (end.crps, end.outbreak_f1), xytext=(4, 5),
                textcoords="offset points", fontsize=8, color=colour,
            )
        gnn = horizon_rows.loc["GNN"]
        axis.scatter(
            gnn.crps, gnn.outbreak_f1, marker="*", s=155, color="#E69F00",
            edgecolor="white", linewidth=.8, zorder=4,
        )
        axis.annotate(
            "GNN only", (gnn.crps, gnn.outbreak_f1), xytext=(5, -12),
            textcoords="offset points", fontsize=8, color="#9A6700",
        )
        axis.set(
            xlabel=f"H{horizon} CRPS  ← better",
            ylabel="Outbreak F1  better →" if horizon == 1 else "",
            title=f"{'ABC'[horizon - 1]}. Horizon {horizon}",
        )
        axis.grid(alpha=.16)
    axes[0].text(
        .02, .98, "Open = no graph\nFilled = + graph attention and gate",
        transform=axes[0].transAxes, va="top", fontsize=8,
        bbox={"facecolor": "white", "alpha": .82, "edgecolor": "none"},
    )
    fig.suptitle(
        "Does graph message passing help? Matched architecture comparisons across horizons"
    )
    _save(fig, output)


def _temporal_stability_figure(predictions, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for model in ("M4", "M6", "M6-INLA"):
        frame = predictions[model]
        frame = frame[frame["horizon"] == 3].copy()
        frame["year"] = frame["target_date"].dt.year
        frame["ae"] = abs(frame["predicted_mean"] - frame["target_cases"]) / frame["population"] * 1e5
        frame["outbreak"] = (frame["target_cases"] > frame["outbreak_threshold"]).astype(float)
        frame["brier"] = (frame["outbreak_probability"] - frame["outbreak"]) ** 2
        annual = frame.groupby("year").agg(mae=("ae", "mean"), brier=("brier", "mean"))
        axes[0].plot(annual.index, annual["mae"], marker="o", color=COLORS[model], label=model)
        axes[1].plot(annual.index, annual["brier"], marker="o", color=COLORS[model], label=model)
    axes[0].set(ylabel="H3 MAE", xlabel="Test year", xticks=(2023, 2024, 2025))
    axes[1].set(ylabel="H3 outbreak Brier score", xlabel="Test year", xticks=(2023, 2024, 2025))
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle("H3 performance across held-out test years")
    _save(fig, output)


def _sparsity_figure(predictions, monthly, output):
    training = monthly[monthly["date"] <= "2020-12-01"]
    zero_rate = training.groupby("fcode")["obs_dengue_cases"].apply(lambda values: (values == 0).mean())
    quartile = pd.qcut(zero_rate, 4, labels=("Q1", "Q2", "Q3", "Q4"), duplicates="drop")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for model in ("M4", "M6"):
        frame = predictions[model]
        frame = frame[frame["horizon"] == 3].copy()
        frame["quartile"] = frame["fcode"].map(quartile)
        frame["ae"] = abs(frame["predicted_mean"] - frame["target_cases"]) / frame["population"] * 1e5
        frame["outbreak"] = (frame["target_cases"] > frame["outbreak_threshold"]).astype(float)
        frame["brier"] = (frame["outbreak_probability"] - frame["outbreak"]) ** 2
        summary = frame.groupby("quartile", observed=True).agg(mae=("ae", "mean"), brier=("brier", "mean"))
        axes[0].plot(summary.index.astype(str), summary["mae"], marker="o", color=COLORS[model], label=model)
        axes[1].plot(summary.index.astype(str), summary["brier"], marker="o", color=COLORS[model], label=model)
    axes[0].set(ylabel="H3 MAE", xlabel="Training zero-case quartile")
    axes[1].set(ylabel="H3 outbreak Brier score", xlabel="Training zero-case quartile")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle("H3 performance as commune observations become sparser")
    _save(fig, output)


def _save(fig, output):
    output = Path(output)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
