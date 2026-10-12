#!/usr/bin/env python3
"""Fit the two fixed deep components for one causal rolling-refit vintage."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_h3.config import load_config, resolve
from dengue_h3.data import build_snapshots, load_monthly, load_outbreak_thresholds
from dengue_h3.graph import build_candidate_graph
from dengue_h3.training import fit_model, predict, save_checkpoint


def fit_component(name, snapshots, graph, train, config, output, draws, representations):
    fitted = fit_model(name, snapshots, graph, train, np.zeros_like(train), config)
    fitted.name = output.name
    output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(fitted, output / "model.pt")
    pd.DataFrame(fitted.history).to_csv(output / "training_history.csv", index=False)
    issue = pd.Timestamp(config["rolling_issue_date"])
    evaluation = np.asarray(snapshots.origins == issue)
    predictions, samples, _ = predict(
        fitted, snapshots, evaluation, draws=draws,
        seed=int(config["sampling"]["seed"]), export_representations=False,
    )
    predictions.to_csv(output / "predictions.csv", index=False)
    np.savez_compressed(output / "predictive_samples.npz", samples=samples)
    if representations:
        selected = train | evaluation
        _, _, latent = predict(
            fitted, snapshots, selected, draws=1,
            seed=int(config["sampling"]["seed"]), export_representations=True,
        )
        latent.insert(
            0, "split",
            np.where(pd.to_datetime(latent["issue_date"]).eq(issue), "test", "train"),
        )
        latent.to_csv(output / "representations.csv", index=False)
    return fitted, len(predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--issue-date", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=33)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--monthly-data", type=Path, required=True)
    parser.add_argument("--climate-columns", nargs="+", required=True)
    args = parser.parse_args()
    root, config = load_config(args.config)
    issue = pd.Timestamp(args.issue_date).to_period("M").to_timestamp()
    data_config = config["data"]
    data_config["monthly_path"] = str(args.monthly_data.resolve())
    data_config["climate_columns"] = list(args.climate_columns)
    monthly = load_monthly(resolve(root, data_config["monthly_path"]), data_config)
    threshold_path = data_config.get("outbreak_thresholds_path")
    resolved_threshold_path = resolve(root, threshold_path) if threshold_path else None
    thresholds = (
        load_outbreak_thresholds(
            resolved_threshold_path,
            unit_column=data_config.get("outbreak_threshold_unit_column", "fcode"),
            threshold_column=data_config.get("outbreak_threshold_column", "epi_threshold"),
        )
        if resolved_threshold_path is not None and resolved_threshold_path.exists()
        else None
    )
    snapshots = build_snapshots(monthly, config, outbreak_thresholds=thresholds)
    graph_config = config["graph"]
    graph = build_candidate_graph(
        snapshots.units, resolve(root, data_config["centroids_path"]),
        resolve(root, data_config["adjacency_path"]),
        k_nearest=int(graph_config["k_nearest"]),
        maximum_distance_km=float(graph_config["maximum_distance_km"]),
        include_administrative_edges=bool(graph_config["include_administrative_edges"]),
    )
    target_dates = pd.to_datetime(snapshots.target_dates.reshape(-1)).to_numpy().reshape(
        snapshots.target_dates.shape
    )
    train_start = np.datetime64("2010-01-01")
    # All three targets of a multi-horizon training example must be observed at issue time.
    train = np.asarray(
        (target_dates[:, -1] >= train_start)
        & (target_dates[:, -1] <= np.datetime64(issue))
    )
    if not train.any() or not np.any(snapshots.origins == issue):
        raise ValueError(f"No causal train/evaluation rows for {issue.date()}")
    common = deepcopy(config)
    common["training"].update({
        "device": args.device, "epochs": args.epochs,
        "fixed_epochs": True, "refit_on_train_validation": False,
    })
    common["sampling"]["draws"] = args.draws
    common["rolling_issue_date"] = str(issue.date())

    base_config = deepcopy(common)
    base_config["model"].update({
        "outbreak_loss_weight": 0.0, "outbreak_positive_weight": 1.0,
        "outbreak_focal_gamma": 0.0, "alert_probability_source": "samples",
    })
    base, base_rows = fit_component(
        "M5", snapshots, graph, train, base_config,
        args.output / "deep", args.draws, True,
    )
    alert_config = deepcopy(common)
    alert_config["model"].update({
        "outbreak_loss_weight": 0.25, "outbreak_positive_weight": "auto",
        "outbreak_focal_gamma": 2.0, "alert_probability_source": "auxiliary",
    })
    alert, alert_rows = fit_component(
        "M5", snapshots, graph, train, alert_config,
        args.output / "peak", args.draws, False,
    )
    manifest = {
        "issue_date": str(issue.date()), "train_start": "2010-01-01",
        "latest_observed_target": str(issue.date()), "training_origins": int(train.sum()),
        "units": len(snapshots.units), "epochs": args.epochs, "draws": args.draws,
        "base_prediction_rows": base_rows, "alert_prediction_rows": alert_rows,
        "base_training_loss": base.history[-1]["training_loss"],
        "alert_training_loss": alert.history[-1]["training_loss"],
    }
    (args.output / "deep_fit_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
