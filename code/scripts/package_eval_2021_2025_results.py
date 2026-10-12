#!/usr/bin/env python3
"""Package the locked 2021--2025 evaluation into one delivery directory."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PLOT_HELPER = Path(__file__).resolve().parent / "make_final_h3_commune_grid.py"


def load_plot_helper():
    spec = importlib.util.spec_from_file_location("h3_grid", PLOT_HELPER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def metric_sources(source: Path) -> dict[str, tuple[str, Path, str]]:
    return {
        "DEEP-LN-GNN": ("deep", source / "DEEP-LN-GNN/metrics.csv", "Deep GNN baseline"),
        "DEEP-NB-GNN": ("deep", source / "DEEP-NB-GNN/metrics.csv", "Negative-binomial GNN"),
        "DEEP-LAND-GNN": (
            "deep",
            source / "DEEP-LAND-GNN/metrics.csv",
            "GNN with climate and land-surface covariates",
        ),
        "DEEP-PEAK-GNN": ("deep", source / "DEEP-PEAK-GNN/metrics.csv", "Peak-sensitive GNN expert"),
        "INLA-TS": (
            "inla",
            source / "INLA-ONLY-test/INLA-TS/metrics.csv",
            "Original temporal-spatial INLA benchmark",
        ),
        "HYBRID-LN-INLA": (
            "deep+inla",
            source / "HYBRID-LN-INLA/test/combined/metrics.csv",
            "Log-normal GNN representation augmented INLA",
        ),
        "HYBRID-NB-INLA": (
            "deep+inla",
            source / "HYBRID-NB-INLA/test/combined/metrics.csv",
            "Negative-binomial GNN representation augmented INLA",
        ),
        "HYBRID-LAND-INLA": (
            "deep+inla",
            source / "HYBRID-LAND-INLA/test/combined/metrics.csv",
            "Land-surface GNN representation augmented INLA",
        ),
        "PROPOSAL-TUNED": (
            "proposal",
            source / "PROPOSAL-PARTIAL-PEAK/metrics.csv",
            "Land-surface GNN representation augmented INLA with partial Peak gate",
        ),
    }


def build_metrics(source: Path, output: Path) -> pd.DataFrame:
    frames = []
    descriptions = []
    for expected_model, (family, path, description) in metric_sources(source).items():
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if set(frame["horizon"].astype(int)) != {1, 2, 3}:
            raise ValueError(f"Missing horizon in {path}")
        if not frame["n"].eq(33_240).all():
            raise ValueError(f"Unexpected test row count in {path}")
        frame["model"] = expected_model
        frame.insert(1, "model_family", family)
        frame.insert(2, "description", description)
        frame.insert(3, "final_figure_model", expected_model in {"INLA-TS", "DEEP-LAND-GNN", "PROPOSAL-TUNED"})
        frames.append(frame)
        descriptions.append(
            {
                "model": expected_model,
                "model_family": family,
                "description": description,
                "final_figure_model": expected_model in {"INLA-TS", "DEEP-LAND-GNN", "PROPOSAL-TUNED"},
            }
        )
    metrics = pd.concat(frames, ignore_index=True).sort_values(["model_family", "model", "horizon"])
    metrics.to_csv(output / "final_metrics.csv", index=False)
    pd.DataFrame(descriptions).to_csv(output / "model_descriptions.csv", index=False)

    summary_columns = [
        "mae",
        "rmse",
        "crps",
        "outbreak_brier",
        "outbreak_precision",
        "outbreak_specificity",
        "outbreak_sensitivity",
        "outbreak_f1",
    ]
    summary = (
        metrics.groupby(["model", "model_family", "description"], as_index=False)[summary_columns]
        .mean()
        .rename(columns={column: f"mean_h1_h3_{column}" for column in summary_columns})
        .sort_values("mean_h1_h3_crps")
    )
    summary.to_csv(output / "model_summary_mean_H1_H3.csv", index=False)
    metrics.loc[metrics["horizon"].eq(3)].to_csv(output / "final_H3_metrics.csv", index=False)
    return metrics


def load_final_predictions(source: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "INLA-TS": source / "INLA-ONLY-test/INLA-TS/predictions.csv",
        "STANDALONE-GNN": source / "DEEP-LAND-GNN/predictions.csv",
        "PROPOSAL": source / "PROPOSAL-PARTIAL-PEAK/predictions.csv",
    }
    predictions = {}
    for model, path in paths.items():
        frame = pd.read_csv(path, dtype={"fcode": str})
        frame["target_date"] = pd.to_datetime(frame["target_date"])
        if set(frame["horizon"].astype(int)) != {1, 2, 3}:
            raise ValueError(f"Missing prediction horizon in {path}")
        for horizon, part in frame.groupby("horizon"):
            if len(part) != 33_240:
                raise ValueError(f"Unexpected H{horizon} row count in {path}")
            dates = part["target_date"]
            if dates.min() != pd.Timestamp("2021-01-01") or dates.max() != pd.Timestamp("2025-12-01"):
                raise ValueError(f"Unexpected test dates in {path}")
        predictions[model] = frame
    return predictions


def select_outbreak_examples(predictions: dict[str, pd.DataFrame], count: int) -> pd.DataFrame:
    h3 = {model: frame.loc[frame["horizon"].eq(3)].copy() for model, frame in predictions.items()}
    rows = []
    for fcode, proposal in h3["PROPOSAL"].groupby("fcode"):
        proposal = proposal.sort_values("target_date")
        observed = proposal["target_cases"].to_numpy(float) > proposal["outbreak_threshold"].to_numpy(float)
        proposal_cutoff = float(proposal["outbreak_decision_threshold"].iloc[0])
        proposal_alert = proposal["outbreak_probability"].to_numpy(float) >= proposal_cutoff
        row = {
            "fcode": str(fcode),
            "observed_outbreaks": int(observed.sum()),
            "proposal_alerts": int(proposal_alert.sum()),
            "proposal_true_positives": int(np.sum(observed & proposal_alert)),
        }
        comparator_tp = []
        for model in ("INLA-TS", "STANDALONE-GNN"):
            part = h3[model].loc[h3[model]["fcode"].eq(str(fcode))].sort_values("target_date")
            cutoff = float(part["outbreak_decision_threshold"].iloc[0]) if "outbreak_decision_threshold" in part else 0.5
            alert = part["outbreak_probability"].to_numpy(float) >= cutoff
            tp = int(np.sum(observed & alert))
            row[f"{model.lower().replace('-', '_')}_true_positives"] = tp
            comparator_tp.append(tp)
        row["proposal_tp_gain_vs_best_comparator"] = row["proposal_true_positives"] - max(comparator_tp)
        rows.append(row)
    selection = (
        pd.DataFrame(rows)
        .loc[lambda frame: frame["proposal_true_positives"].gt(0)]
        .sort_values(
            ["proposal_tp_gain_vs_best_comparator", "proposal_true_positives", "observed_outbreaks"],
            ascending=False,
        )
        .head(count)
        .reset_index(drop=True)
    )
    selection.insert(0, "selection_rank", np.arange(1, len(selection) + 1))
    selection["selection_rule"] = (
        "descriptive H3 examples ranked by proposal true-positive gain, then proposal true positives"
    )
    selection["burden_cutoff"] = np.nan
    return selection


def build_figure(source: Path, output: Path, predictions: dict[str, pd.DataFrame], count: int) -> pd.DataFrame:
    helper = load_plot_helper()
    helper.MODELS = ("INLA-TS", "STANDALONE-GNN", "PROPOSAL")
    helper.DISPLAY_NAMES = {
        "INLA-TS": "INLA–TS",
        "STANDALONE-GNN": "Standalone land-surface GNN",
        "PROPOSAL": "Proposal: representation-INLA + partial Peak gate",
    }
    helper.COLORS = {"INLA-TS": "#6B7280", "STANDALONE-GNN": "#009E73", "PROPOSAL": "#0072B2"}
    h3 = {model: frame.loc[frame["horizon"].eq(3)].copy() for model, frame in predictions.items()}
    selection = select_outbreak_examples(predictions, count)
    names_path = ROOT.parent / "dengue_forecast_v1/dengue_training/data/static/commune_names.csv"
    names = helper.load_names(names_path)
    selection["region_name"] = selection["fcode"].map(lambda value: names.get(str(value), (str(value), ""))[0])
    selection["province_name"] = selection["fcode"].map(lambda value: names.get(str(value), (str(value), ""))[1])
    selection.to_csv(output / "Fig_H3_predicted_outbreak_examples_selection.csv", index=False)
    figure_base = output / "Fig_H3_predicted_outbreak_examples"
    helper.plot_grid(
        h3,
        selection,
        names,
        helper.overall_outbreak_metrics(h3),
        figure_base,
        "Locked 2021–2025 H3 outbreak forecasts",
    )
    pdf = figure_base.with_suffix(".pdf")
    if pdf.exists():
        pdf.unlink()
    return selection


def copy_final_predictions(source: Path, output: Path) -> None:
    destination = output / "predictions"
    destination.mkdir(parents=True, exist_ok=True)
    sources = {
        "INLA_TS_predictions.csv": source / "INLA-ONLY-test/INLA-TS/predictions.csv",
        "standalone_LAND_GNN_predictions.csv": source / "DEEP-LAND-GNN/predictions.csv",
        "proposal_predictions.csv": source / "PROPOSAL-PARTIAL-PEAK/predictions.csv",
    }
    for name, path in sources.items():
        shutil.copy2(path, destination / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "results/eval_2021_2025_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/eval_2021_2025_final")
    parser.add_argument("--commune-count", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics(args.source, args.output)
    predictions = load_final_predictions(args.source)
    selection = build_figure(args.source, args.output, predictions, args.commune_count)
    copy_final_predictions(args.source, args.output)
    shutil.copy2(args.source / "PROPOSAL-PARTIAL-PEAK/selection.json", args.output / "proposal_validation_selection.json")
    shutil.copy2(args.source / "selected_hybrid.txt", args.output / "selected_hybrid.txt")
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "development_train": "2010-01 through 2018-12",
        "blocked_validation": "2019-01 through 2020-12",
        "locked_test": "2021-01 through 2025-12",
        "test_rows_per_horizon": 33240,
        "test_months_per_horizon": 60,
        "communes": 554,
        "metric_models": metrics["model"].drop_duplicates().tolist(),
        "figure_communes": selection["fcode"].tolist(),
        "source_results": str(args.source.resolve()),
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Packaged final results in {args.output.resolve()}")


if __name__ == "__main__":
    main()
