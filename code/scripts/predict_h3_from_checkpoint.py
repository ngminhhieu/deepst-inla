#!/usr/bin/env python3
"""Run H3 inference from a saved neural checkpoint without retraining."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dengue_h3.config import load_config, resolve
from dengue_h3.data import build_snapshots, load_monthly, load_outbreak_thresholds
from dengue_h3.graph import build_candidate_graph
from dengue_h3.models import DengueForecastNet
from dengue_h3.specs import get_spec
from dengue_h3.training import FittedModel, predict


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return f"external:{resolved.name}"


def load_checkpoint(path: Path) -> dict:
    # Checkpoints written with NumPy 2 reference numpy._core; this runtime uses
    # NumPy 1.x.  The alias is safe because the relevant ndarray reconstructors
    # are identical.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representations", action="store_true")
    parser.add_argument("--draws", type=int, default=500)
    args = parser.parse_args()

    root, config = load_config(args.config)
    data_cfg = config["data"]
    monthly = load_monthly(resolve(root, data_cfg["monthly_path"]), data_cfg)
    thresholds = load_outbreak_thresholds(
        resolve(root, data_cfg["outbreak_thresholds_path"]),
        unit_column=data_cfg.get("outbreak_threshold_unit_column", "fcode"),
        threshold_column=data_cfg["outbreak_threshold_column"],
    )
    snapshots = build_snapshots(monthly, config, outbreak_thresholds=thresholds)
    graph_cfg = config["graph"]
    graph = build_candidate_graph(
        snapshots.units,
        resolve(root, data_cfg["centroids_path"]),
        resolve(root, data_cfg["adjacency_path"]),
        k_nearest=int(graph_cfg["k_nearest"]),
        maximum_distance_km=float(graph_cfg["maximum_distance_km"]),
        include_administrative_edges=bool(graph_cfg["include_administrative_edges"]),
    )

    saved = load_checkpoint(args.checkpoint)
    model_cfg = config["model"]
    network = DengueForecastNet(
        spec=get_spec("M5"),
        local_feature_size=snapshots.local.shape[-1],
        climate_feature_size=snapshots.climate.shape[-1],
        seasonal_feature_size=snapshots.seasonal.shape[-1],
        node_count=len(snapshots.units),
        horizon_count=len(snapshots.horizons),
        hidden_size=int(model_cfg["hidden_size"]),
        commune_embedding_size=int(model_cfg["commune_embedding_size"]),
        dilations=tuple(int(value) for value in model_cfg["dilations"]),
        kernel_size=int(model_cfg["kernel_size"]),
        dropout=float(model_cfg["dropout"]),
        minimum_dispersion=float(model_cfg["minimum_dispersion"]),
        likelihood=str(saved["likelihood"]),
        maximum_log_sigma=float(model_cfg["maximum_log_sigma"]),
        candidate_mask=torch.from_numpy(graph.mask),
        edge_features=torch.from_numpy(graph.edge_features),
        fixed_weights=torch.from_numpy(graph.fixed_weights),
        graph_heads=int(model_cfg["graph_heads"]),
    )
    network.load_state_dict(saved["state_dict"])
    network.eval()
    fitted = FittedModel(
        name=str(saved["model_name"]), network=network,
        local_scale=saved["local_scale"], climate_scale=saved["climate_scale"],
        seasonal_scale=saved["seasonal_scale"], history=saved["history"],
        likelihood=str(saved["likelihood"]),
        outbreak_loss_weight=float(saved.get("outbreak_loss_weight", 0.0)),
        outbreak_positive_weight=saved.get("outbreak_positive_weight"),
        outbreak_focal_gamma=float(saved.get("outbreak_focal_gamma", 0.0)),
        alert_probability_source=str(saved.get("alert_probability_source", "samples")),
    )

    origins = pd.DatetimeIndex(snapshots.origins)
    test_mask = np.asarray(origins.to_period("M").isin(pd.period_range("2026-01", "2026-08", freq="M")))
    test_predictions, test_cube, test_representations = predict(
        fitted, snapshots, test_mask, draws=args.draws, seed=1234,
        export_representations=args.representations,
    )
    h3 = max(snapshots.horizons)
    h3_index = list(snapshots.horizons).index(h3)
    test_h3 = test_predictions.loc[test_predictions["horizon"].eq(h3)].reset_index(drop=True)
    test_samples = test_cube[:, :, h3_index, :].reshape(-1, args.draws)
    if len(test_h3) != len(test_samples):
        raise ValueError("H3 prediction/sample mismatch")

    args.output.mkdir(parents=True, exist_ok=True)
    test_h3.to_csv(args.output / "predictions_h3.csv", index=False)
    np.savez_compressed(args.output / "predictive_samples_h3.npz", samples=test_samples)

    if args.representations:
        development_mask = np.asarray(origins.to_period("M").isin(
            pd.period_range("2010-01", "2020-09", freq="M")
        ))
        _, _, development = predict(
            fitted, snapshots, development_mask, draws=1, seed=1234,
            export_representations=True,
        )
        development = development.loc[development["horizon"].eq(h3)].copy()
        target = pd.to_datetime(development["target_date"])
        development.insert(0, "split", np.where(
            target.le(pd.Timestamp("2018-12-01")), "train", "validation"
        ))
        test_representations = test_representations.loc[
            test_representations["horizon"].eq(h3)
        ].copy()
        test_representations.insert(0, "split", "test")
        representations = pd.concat([development, test_representations], ignore_index=True)
        representations.to_csv(args.output / "representations_h3.csv", index=False)

    manifest = {
        "checkpoint": portable_path(args.checkpoint),
        "checkpoint_training": "2010-2018 selection; refit through 2020",
        "forecast_horizon": 3,
        "issue_months": ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"],
        "target_months": ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10", "2026-11"],
        "communes": len(snapshots.units), "draws": args.draws,
        "rows": len(test_h3), "retrained": False,
    }
    (args.output / "inference_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
