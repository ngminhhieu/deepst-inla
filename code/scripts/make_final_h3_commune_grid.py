#!/usr/bin/env python3
"""H3 trajectory grid: communes by scientifically distinct final models."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

mpl.rcParams.update({
    "font.size": 12.5,
    "axes.titlesize": 13.5,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 11.5,
})
MODELS = ("INLA-TS", "STANDALONE-GNN", "PROPOSAL")
PREDICTION_PATHS = {
    "INLA-TS": Path("inla_test_exact/INLA-TS/predictions.csv"),
    "STANDALONE-GNN": Path("REPR-GNN/predictions.csv"),
    "PROPOSAL": Path(
        "proposal_tuned/PROPOSAL-TUNED/predictions.csv"
    ),
}
DISPLAY_NAMES = {
    "INLA-TS": "INLA–TS",
    "STANDALONE-GNN": "Standalone GNN",
    "PROPOSAL": "Tuned proposal: representation-INLA + partial Peak gate",
}
COLORS = {
    "INLA-TS": "#6B7280",
    "STANDALONE-GNN": "#009E73",
    "PROPOSAL": "#0072B2",
}


def load_predictions(results_root: Path) -> dict[str, pd.DataFrame]:
    output = {}
    for model, relative_path in PREDICTION_PATHS.items():
        path = results_root / relative_path
        if not path.exists():
            raise SystemExit(f"Required prediction file is missing: {path}")
        frame = pd.read_csv(path, dtype={"fcode": str})
        frame["target_date"] = pd.to_datetime(frame["target_date"])
        output[model] = frame[frame["horizon"].eq(3)].copy()
    return output


def overall_outbreak_metrics(
    predictions: dict[str, pd.DataFrame]
) -> dict[str, dict[str, float]]:
    metrics = {}
    for model, frame in predictions.items():
        observed = frame["target_cases"].to_numpy(float) > frame[
            "outbreak_threshold"
        ].to_numpy(float)
        decision_threshold = (
            float(frame["outbreak_decision_threshold"].iloc[0])
            if "outbreak_decision_threshold" in frame
            else 0.5
        )
        predicted = frame["outbreak_probability"].to_numpy(float) >= decision_threshold
        tp = int(np.sum(observed & predicted))
        fp = int(np.sum(~observed & predicted))
        fn = int(np.sum(observed & ~predicted))
        tn = int(np.sum(~observed & ~predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        metrics[model] = {
            "precision": precision,
            "specificity": specificity,
            "recall": recall,
            "sensitivity": recall,
            "f1": f1,
        }
    return metrics


def load_names(path: Path) -> dict[str, tuple[str, str]]:
    frame = pd.read_csv(path, dtype={"fcode": str})
    required = {"fcode", "region_name", "province_name"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Name file must contain {sorted(required)}")
    names = {
        str(row.fcode): (str(row.region_name), str(row.province_name))
        for row in frame.itertuples(index=False)
    }
    names.update(
        {
            "30781": ("Xã Tiên Hải", "Tỉnh An Giang"),
            "30814": ("Xã Hòn Nghệ", "Tỉnh An Giang"),
            "31078": ("Đặc khu Phú Quốc", "Tỉnh An Giang"),
            "31105": ("Đặc khu Thổ Châu", "Tỉnh An Giang"),
        }
    )
    return names


def select_communes(
    predictions: dict[str, pd.DataFrame], count: int
) -> pd.DataFrame:
    """Rank by forecast disagreement, not by realised model error."""
    forecasts = []
    observed = None
    for model, frame in predictions.items():
        frame = frame.copy()
        frame["predicted_incidence"] = (
            frame["predicted_mean"] / frame["population"] * 100_000
        )
        frame["observed_incidence"] = (
            frame["target_cases"] / frame["population"] * 100_000
        )
        index = pd.MultiIndex.from_arrays(
            [frame["fcode"], frame["target_date"]],
            names=("fcode", "target_date"),
        )
        forecasts.append(
            pd.Series(
                frame["predicted_incidence"].to_numpy(), index=index, name=model
            )
        )
        if observed is None:
            observed = pd.Series(
                frame["observed_incidence"].to_numpy(), index=index, name="observed"
            )
    forecast_frame = pd.concat(forecasts, axis=1)
    divergence = forecast_frame.std(axis=1).groupby(level=0).mean()
    burden = observed.groupby(level=0).mean()
    burden_cutoff = float(burden.quantile(0.50))
    eligible = burden[burden >= burden_cutoff].index
    ranked = (
        pd.DataFrame(
            {
                "mean_forecast_sd": divergence.loc[eligible],
                "mean_observed_incidence": burden.loc[eligible],
            }
        )
        .sort_values("mean_forecast_sd", ascending=False)
        .head(count)
        .reset_index()
    )
    ranked["selection_rule"] = (
        "top H3 mean forecast SD among communes at/above median observed incidence"
    )
    ranked["burden_cutoff"] = burden_cutoff
    ranked.insert(0, "selection_rank", np.arange(1, len(ranked) + 1))
    return ranked


def describe_communes(
    predictions: dict[str, pd.DataFrame], communes: list[str]
) -> pd.DataFrame:
    """Describe an explicit commune set without selecting on realised errors."""
    forecasts = []
    observed = None
    for model, frame in predictions.items():
        part = frame[frame["fcode"].isin(communes)].copy()
        part["predicted_incidence"] = (
            part["predicted_mean"] / part["population"] * 100_000
        )
        part["observed_incidence"] = (
            part["target_cases"] / part["population"] * 100_000
        )
        index = pd.MultiIndex.from_arrays(
            [part["fcode"], part["target_date"]],
            names=("fcode", "target_date"),
        )
        forecasts.append(
            pd.Series(part["predicted_incidence"].to_numpy(), index=index, name=model)
        )
        if observed is None:
            observed = pd.Series(
                part["observed_incidence"].to_numpy(), index=index, name="observed"
            )
    forecast_frame = pd.concat(forecasts, axis=1)
    divergence = forecast_frame.std(axis=1).groupby(level=0).mean()
    burden = observed.groupby(level=0).mean()
    missing = [unit for unit in communes if unit not in divergence.index]
    if missing:
        raise ValueError(f"No H3 predictions for commune(s): {', '.join(missing)}")
    return pd.DataFrame(
        {
            "selection_rank": np.arange(1, len(communes) + 1),
            "fcode": communes,
            "mean_forecast_sd": [divergence.loc[unit] for unit in communes],
            "mean_observed_incidence": [burden.loc[unit] for unit in communes],
            "selection_rule": "user-specified commune list (order preserved)",
            "burden_cutoff": np.nan,
        }
    )


def select_proposal_gain(
    predictions: dict[str, pd.DataFrame], count: int, candidates: list[str] | None
) -> pd.DataFrame:
    """Select illustrative test cases transparently by proposal MAE gain."""
    rows = []
    common = set.intersection(
        *(set(frame["fcode"].astype(str)) for frame in predictions.values())
    )
    if candidates is not None:
        missing = sorted(set(candidates).difference(common))
        if missing:
            raise ValueError(f"No H3 predictions for commune(s): {', '.join(missing)}")
        common &= set(candidates)
    for unit in common:
        errors = {}
        observed_burden = np.nan
        for model, frame in predictions.items():
            part = frame[frame["fcode"].eq(unit)]
            scale = 100_000 / part["population"].to_numpy(float)
            errors[model] = float(
                np.mean(np.abs(part["predicted_mean"] - part["target_cases"]) * scale)
            )
            observed_burden = float(np.mean(part["target_cases"].to_numpy(float) * scale))
        gain = min(errors["INLA-TS"], errors["STANDALONE-GNN"]) - errors["PROPOSAL"]
        rows.append(
            {
                "fcode": unit,
                "proposal_mae_incidence": errors["PROPOSAL"],
                "inla_mae_incidence": errors["INLA-TS"],
                "gnn_mae_incidence": errors["STANDALONE-GNN"],
                "proposal_gain_vs_best_comparator": gain,
                "mean_observed_incidence": observed_burden,
            }
        )
    selected = (
        pd.DataFrame(rows)
        .sort_values("proposal_gain_vs_best_comparator", ascending=False)
        .head(count)
        .reset_index(drop=True)
    )
    selected.insert(0, "selection_rank", np.arange(1, len(selected) + 1))
    selected["selection_rule"] = (
        "illustrative test cases ranked by H3 proposal incidence-MAE gain over the better comparator"
    )
    selected["burden_cutoff"] = np.nan
    return selected


def plot_grid(
    predictions: dict[str, pd.DataFrame],
    selection: pd.DataFrame,
    names: dict[str, tuple[str, str]],
    model_metrics: dict[str, dict[str, float]],
    output: Path,
    figure_title: str | None = None,
) -> None:
    fig, axes = plt.subplots(
        len(selection),
        len(MODELS),
        figsize=(max(14.8, 3.7 * len(MODELS)), max(9.4, 2.7 * len(selection) + 1.2)),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    commune_count = predictions[MODELS[0]]["fcode"].nunique()
    for row, unit in enumerate(selection["fcode"].astype(str)):
        commune, province = names.get(unit, (unit, ""))
        for column, model in enumerate(MODELS):
            axis = axes[row, column]
            part = predictions[model][predictions[model]["fcode"].eq(unit)].sort_values(
                "target_date"
            )
            if part.empty:
                raise ValueError(f"No H3 predictions for commune {unit}: {model}")
            scale = 100_000 / part["population"].to_numpy(float)
            date = part["target_date"].to_numpy()
            observed = part["target_cases"].to_numpy(float) * scale
            forecast = part["predicted_mean"].to_numpy(float) * scale
            threshold = part["outbreak_threshold"].to_numpy(float) * scale
            axis.fill_between(
                date,
                part["lower_80"].to_numpy(float) * scale,
                part["upper_80"].to_numpy(float) * scale,
                color=COLORS[model],
                alpha=0.27,
                label="80% PI",
            )
            axis.plot(
                date,
                forecast,
                color=COLORS[model],
                linewidth=2.0,
                label="Mean forecast",
            )
            axis.plot(
                date,
                observed,
                color="#111111",
                linewidth=1.5,
                marker="o",
                markersize=2.5,
                label="Observed",
            )
            axis.plot(
                date,
                threshold,
                color="#D55E00",
                linestyle=":",
                linewidth=1.2,
                alpha=0.85,
                label="Outbreak threshold",
            )
            outbreak = np.isfinite(threshold) & (observed > threshold)
            decision_threshold = (
                float(part["outbreak_decision_threshold"].iloc[0])
                if "outbreak_decision_threshold" in part
                else 0.5
            )
            predicted_outbreak = (
                part["outbreak_probability"].to_numpy(float) >= decision_threshold
            )
            axis.scatter(
                date[outbreak],
                observed[outbreak],
                color="#D62728",
                s=19,
                zorder=4,
                label="Observed outbreak",
            )
            axis.scatter(
                date[predicted_outbreak],
                forecast[predicted_outbreak],
                marker="^",
                facecolor="#CC79A7",
                edgecolor="white",
                linewidth=0.7,
                s=50,
                zorder=5,
                label="Predicted outbreak alert",
            )
            commune_mae = float(np.mean(np.abs(forecast - observed)))
            true_positive = int(np.sum(outbreak & predicted_outbreak))
            alert_count = int(np.sum(predicted_outbreak))
            axis.text(
                0.98,
                0.72 if row == 0 else 0.96,
                f"MAE={commune_mae:.1f} | Alerts={alert_count} (TP={true_positive})",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=10.7,
                bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.82},
            )
            if row == 0:
                axis.set_title(DISPLAY_NAMES[model], fontsize=13.5, fontweight="bold")
                metric = model_metrics[model]
                axis.text(
                    0.02,
                    0.97,
                    f"All {commune_count} communes, H3\n"
                    f"Precision={metric['precision']:.3f} | "
                    f"Specificity={metric['specificity']:.3f}\n"
                    f"Sensitivity={metric['sensitivity']:.3f} | "
                    f"F1={metric['f1']:.3f}",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=11,
                    bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.88},
                )
            if column == 0:
                location = f"{commune}\n{province}" if province else commune
                axis.set_ylabel(f"{location}\nIncidence / 100,000")
            if row == len(selection) - 1:
                axis.set_xlabel("H3 target month")
                axis.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                axis.tick_params(axis="x", rotation=35)
            axis.grid(alpha=0.18)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=6,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.025),
    )
    rule = selection["selection_rule"].iloc[0]
    if figure_title:
        title = figure_title
    elif rule.startswith("user-specified"):
        title = "H3 forecasts for user-specified communes"
    elif rule.startswith("illustrative"):
        title = (
            "Illustrative H3 forecasts with largest proposal gain\n"
            "Selection is descriptive; overall H3 outbreak metrics are shown in each column"
        )
    else:
        title = (
            "H3 forecasts for communes with greatest disagreement between final models\n"
            "Rows selected by forecast disagreement among communes with observed burden at or above the median"
        )
    fig.suptitle(title, fontsize=16.5, y=1.065, linespacing=1.55)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--names-path",
        type=Path,
        default=ROOT.parent
        / "dengue_forecast_v1/dengue_training/data/static/commune_names.csv",
    )
    parser.add_argument("--commune-count", type=int, default=3)
    parser.add_argument(
        "--communes",
        help="Comma-separated fcodes to plot in the given order; overrides --commune-count",
    )
    parser.add_argument(
        "--selection-rule",
        choices=("disagreement", "proposal-gain"),
        default="disagreement",
    )
    parser.add_argument(
        "--candidate-communes",
        help="Optional comma-separated candidate pool for --selection-rule proposal-gain",
    )
    parser.add_argument(
        "--results-root", type=Path, default=ROOT / "results/qc_v1"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper/figures/Fig_final_H3_three_communes",
    )
    parser.add_argument(
        "--figure-title",
        help="Optional explicit title for the figure.",
    )
    args = parser.parse_args()
    predictions = load_predictions(args.results_root)
    if args.communes:
        communes = [value.strip().replace(" ", "") for value in args.communes.split(",")]
        communes = [value for value in communes if value]
        selection = describe_communes(predictions, communes)
    elif args.selection_rule == "proposal-gain":
        candidates = None
        if args.candidate_communes:
            candidates = [
                value.strip().replace(" ", "")
                for value in args.candidate_communes.split(",")
                if value.strip()
            ]
        selection = select_proposal_gain(predictions, args.commune_count, candidates)
    else:
        selection = select_communes(predictions, args.commune_count)
    names = load_names(args.names_path)
    selection["region_name"] = selection["fcode"].map(
        lambda value: names.get(str(value), (str(value), ""))[0]
    )
    selection["province_name"] = selection["fcode"].map(
        lambda value: names.get(str(value), (str(value), ""))[1]
    )
    selection_path = args.output.with_name(args.output.name + "_selection.csv")
    selection.to_csv(selection_path, index=False)
    plot_grid(
        predictions,
        selection,
        names,
        overall_outbreak_metrics(predictions),
        args.output,
        args.figure_title,
    )
    print(f"wrote {args.output.with_suffix('.png')} and .pdf")
    print(selection[["selection_rank", "fcode", "region_name", "province_name"]].to_string(index=False))


if __name__ == "__main__":
    main()
