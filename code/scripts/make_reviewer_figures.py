#!/usr/bin/env python3
"""Generate the paper's H1--H3 deep-model trajectory comparison."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ("CNN", "LSTM", "CNN-LSTM", "GNN", "LSTM-GNN", "M6")
DISPLAY_NAMES = {
    "CNN": "CNN",
    "LSTM": "LSTM",
    "CNN-LSTM": "CNN–LSTM",
    "GNN": "GNN",
    "LSTM-GNN": "LSTM–GNN",
    "M6": "TCN–GNN",
}
COLORS = {
    "CNN": "#7A7A7A",
    "LSTM": "#009E73",
    "CNN-LSTM": "#CC79A7",
    "GNN": "#E69F00",
    "LSTM-GNN": "#56B4E9",
    "M6": "#0072B2",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument("--paper-dir", default="../paper")
    parser.add_argument("--names-path", required=True)
    parser.add_argument(
        "--grid-commune", default="auto",
        help="Commune code, or 'auto' for the commune with greatest model divergence",
    )
    args = parser.parse_args()

    results = Path(args.results_dir).resolve()
    paper = Path(args.paper_dir).resolve()
    predictions = {}
    for model in MODELS:
        path = results / model / "predictions.csv"
        if not path.exists():
            raise SystemExit(f"Required result is missing: {path}")
        frame = pd.read_csv(path, dtype={"fcode": str})
        frame["target_date"] = pd.to_datetime(frame["target_date"])
        predictions[model] = frame

    names = _load_names(Path(args.names_path))
    unit = (
        _select_divergent_commune(predictions)
        if str(args.grid_commune).lower() == "auto"
        else str(args.grid_commune)
    )
    if unit not in set(predictions[MODELS[0]]["fcode"]):
        raise SystemExit(f"Commune {unit} is absent from predictions")
    _model_horizon_grid(
        predictions,
        unit,
        names.get(unit, (unit, "")),
        paper / "figures" / "Fig6_model_horizon_grid",
    )
    print(f"wrote H1--H3 trajectory grid for commune {unit}")


def _model_horizon_grid(predictions, unit, place, output):
    fig, axes = plt.subplots(
        3, len(MODELS), figsize=(18, 9.2), sharex=True, sharey="row",
        constrained_layout=True,
    )
    for column, model in enumerate(MODELS):
        unit_frame = predictions[model][predictions[model]["fcode"] == unit]
        for row, horizon in enumerate((1, 2, 3)):
            axis = axes[row, column]
            part = unit_frame[unit_frame["horizon"] == horizon].sort_values("target_date")
            population = part["population"].to_numpy(float)
            scale = 100_000 / population
            date = part["target_date"].to_numpy()
            observed = part["target_cases"].to_numpy(float) * scale
            threshold = part["outbreak_threshold"].to_numpy(float) * scale
            axis.fill_between(
                date,
                part["lower_95"].to_numpy(float) * scale,
                part["upper_95"].to_numpy(float) * scale,
                color=COLORS[model], alpha=.12, label="95% PI",
            )
            axis.fill_between(
                date,
                part["lower_80"].to_numpy(float) * scale,
                part["upper_80"].to_numpy(float) * scale,
                color=COLORS[model], alpha=.26, label="80% PI",
            )
            axis.plot(
                date, part["predicted_mean"].to_numpy(float) * scale,
                color=COLORS[model], linewidth=1.8, label="Mean forecast",
            )
            axis.plot(
                date, observed, color="#111111", linewidth=1.45,
                marker="o", markersize=2.3, label="Observed",
            )
            axis.plot(
                date, threshold, color="#D55E00", linestyle=":",
                linewidth=1.1, alpha=.85, label="Outbreak threshold",
            )
            outbreak = np.isfinite(threshold) & (observed > threshold)
            axis.scatter(
                date[outbreak], observed[outbreak], color="#D62728",
                s=17, zorder=4, label="Observed outbreak",
            )
            if row == 0:
                axis.set_title(DISPLAY_NAMES[model], fontsize=11, fontweight="bold")
            if column == 0:
                axis.set_ylabel(f"H{horizon}\nIncidence / 100,000")
            if row == 2:
                axis.set_xlabel("Target month")
                axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                axis.tick_params(axis="x", rotation=35)
            axis.grid(alpha=.18)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, frameon=False, loc="lower center",
               bbox_to_anchor=(.5, -.025))
    commune, province = place
    location = f"{commune}, {province}" if province else commune
    fig.suptitle(
        f"Held-out deep-learning forecasts — {location}",
        fontsize=14, y=1.025,
    )
    _save(fig, output)


def _select_divergent_commune(predictions):
    """Select a data-rich commune where model means visibly disagree."""
    series = []
    observed = None
    for model, frame in predictions.items():
        part = frame[frame["horizon"] == 3].copy()
        part["predicted_incidence"] = (
            part["predicted_mean"] / part["population"] * 100_000
        )
        part["observed_incidence"] = (
            part["target_cases"] / part["population"] * 100_000
        )
        index = pd.MultiIndex.from_arrays(
            [part["fcode"].astype(str), part["target_date"]],
            names=("fcode", "target_date"),
        )
        series.append(pd.Series(part["predicted_incidence"].to_numpy(), index=index, name=model))
        if observed is None:
            observed = pd.Series(
                part["observed_incidence"].to_numpy(), index=index, name="observed"
            )
    forecasts = pd.concat(series, axis=1)
    score = forecasts.std(axis=1).groupby(level=0).mean()
    burden = observed.groupby(level=0).mean()
    eligible = burden[burden >= burden.quantile(.50)].index
    score = score.loc[score.index.intersection(eligible)]
    return str(score.idxmax())


def _load_names(path):
    frame = pd.read_csv(path, dtype={"fcode": str})
    required = {"fcode", "region_name", "province_name"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Name file must contain {sorted(required)}")
    names = {
        str(row.fcode): (str(row.region_name), str(row.province_name))
        for row in frame.itertuples(index=False)
    }
    names.update({
        "30781": ("Xã Tiên Hải", "Tỉnh An Giang"),
        "30814": ("Xã Hòn Nghệ", "Tỉnh An Giang"),
        "31078": ("Đặc khu Phú Quốc", "Tỉnh An Giang"),
        "31105": ("Đặc khu Thổ Châu", "Tỉnh An Giang"),
    })
    return names


def _save(fig, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
