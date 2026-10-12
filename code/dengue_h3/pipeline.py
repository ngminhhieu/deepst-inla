from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve
from .data import (
    build_snapshots,
    load_future_covariates,
    load_monthly,
    load_outbreak_thresholds,
    split_mask,
)
from .evaluation import score_predictions
from .graph import build_candidate_graph
from .graph import CandidateGraph
from .training import fit_model, predict, save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _portable_manifest_path(path: Path) -> str:
    """Record a reproducible relative label without leaking a workstation path."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return f"external_input/{resolved.name}"


def _predict_target_window(
    fitted,
    snapshots,
    start: str,
    end: str,
    *,
    draws: int,
    seed: int,
    export_representations: bool,
    require_equal_samples: bool = True,
):
    """Predict an exact target-date window independently for every horizon."""
    target_dates = pd.to_datetime(snapshots.target_dates.reshape(-1)).to_numpy().reshape(
        snapshots.target_dates.shape
    )
    lower = np.datetime64(pd.Timestamp(start))
    upper = np.datetime64(pd.Timestamp(end))
    in_window = (target_dates >= lower) & (target_dates <= upper)
    origin_mask = in_window.any(axis=1)
    predictions, samples, representations = predict(
        fitted,
        snapshots,
        origin_mask,
        draws=draws,
        seed=seed,
        export_representations=export_representations,
    )
    selected_dates = target_dates[origin_mask]
    horizon_samples = []
    expected_origins = None
    for horizon_index in range(len(snapshots.horizons)):
        selected = (selected_dates[:, horizon_index] >= lower) & (
            selected_dates[:, horizon_index] <= upper
        )
        count = int(selected.sum())
        if expected_origins is None:
            expected_origins = count
        elif count != expected_origins:
            if require_equal_samples:
                raise ValueError(
                    "Exact target window has unequal origin counts by horizon"
                )
            filtered_samples = samples
            break
        horizon_samples.append(samples[selected, :, horizon_index, :])
    else:
        filtered_samples = np.stack(horizon_samples, axis=2)
    prediction_dates = pd.to_datetime(predictions["target_date"])
    predictions = predictions[
        prediction_dates.between(pd.Timestamp(start), pd.Timestamp(end))
    ].reset_index(drop=True)
    if export_representations:
        representation_dates = pd.to_datetime(representations["target_date"])
        representations = representations[
            representation_dates.between(pd.Timestamp(start), pd.Timestamp(end))
        ].reset_index(drop=True)
    return predictions, filtered_samples, representations


def run_experiment(
    root: Path,
    config: dict,
    model_name: str,
    output_name: str | None = None,
    *,
    results_root: Path | None = None,
    export_representations: bool | None = None,
    export_validation_predictions: bool = False,
) -> Path:
    data_config = config["data"]
    monthly = load_monthly(resolve(root, data_config["monthly_path"]), data_config)
    future_covariates = None
    future_columns = list(data_config.get("future_covariate_columns", []))
    future_path = data_config.get("future_covariates_path")
    if future_columns:
        if not future_path:
            raise ValueError("future_covariate_columns require data.future_covariates_path")
        future_covariates = load_future_covariates(
            resolve(root, future_path),
            future_columns,
            unit_column=str(data_config.get("future_unit_column", "fcode")),
            issue_column=str(data_config.get("future_issue_column", "issue_date")),
            target_column=str(data_config.get("future_target_column", "target_date")),
        )
    threshold_path = data_config.get("outbreak_thresholds_path")
    outbreak_thresholds = None
    if threshold_path and resolve(root, threshold_path).exists():
        outbreak_thresholds = load_outbreak_thresholds(
            resolve(root, threshold_path),
            unit_column=str(data_config.get("outbreak_threshold_unit_column", "fcode")),
            threshold_column=str(
                data_config.get("outbreak_threshold_column", "epi_threshold")
            ),
        )
    snapshots = build_snapshots(
        monthly,
        config,
        future_covariates=future_covariates,
        outbreak_thresholds=outbreak_thresholds,
    )
    graph_config = config["graph"]
    graph = build_candidate_graph(
        snapshots.units,
        resolve(root, data_config["centroids_path"]),
        resolve(root, data_config["adjacency_path"]),
        k_nearest=int(graph_config["k_nearest"]),
        maximum_distance_km=float(graph_config["maximum_distance_km"]),
        include_administrative_edges=bool(
            graph_config["include_administrative_edges"]
        ),
    )
    excluded_units = {
        str(value) for value in config["training"].get("excluded_units", [])
    }
    if excluded_units:
        unknown = excluded_units.difference(snapshots.units)
        if unknown:
            raise ValueError(f"Unknown excluded communes: {sorted(unknown)[:10]}")
        held_out = np.asarray(
            [unit in excluded_units for unit in snapshots.units], dtype=bool
        )
        training = ~held_out
        # destination x source: training destinations cannot consume held-out
        # histories. Held-out destinations may still attend training neighbours.
        mask = graph.mask.copy()
        mask[np.ix_(training, held_out)] = False
        edge_features = graph.edge_features.copy()
        edge_features[~mask] = 0.0
        fixed_weights = graph.fixed_weights.copy()
        fixed_weights[~mask] = 0.0
        fixed_weights /= fixed_weights.sum(axis=1, keepdims=True).clip(min=1e-8)
        graph = CandidateGraph(
            units=graph.units,
            mask=mask,
            edge_features=edge_features,
            fixed_weights=fixed_weights,
        )
    split = config["split"]
    train = split_mask(snapshots, split["train_start"], split["train_end"])
    validation = split_mask(
        snapshots, split["validation_start"], split["validation_end"]
    )
    test = split_mask(snapshots, split["test_start"], split["test_end"])
    fitted = fit_model(model_name, snapshots, graph, train, validation, config)
    result_name = output_name or model_name.upper()
    fitted.name = result_name
    output_dir = (results_root or root.parent / "results") / result_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(fitted, output_dir / "model.pt")
    pd.DataFrame(fitted.history).to_csv(output_dir / "training_history.csv", index=False)
    if fitted.selection_history is not None:
        pd.DataFrame(fitted.selection_history).to_csv(
            output_dir / "selection_history.csv", index=False
        )

    # Representations for INLA are exported for the training, validation, and test
    # windows. The split label is explicit so R cannot accidentally fit on test rows.
    if export_representations is None:
        export_representations = model_name.upper() == "M6"
    representation_frames = []
    selected_splits = (
        (("train", train), ("validation", validation), ("test", test))
        if export_representations
        else (
            (("validation", validation), ("test", test))
            if export_validation_predictions
            else (("test", test),)
        )
    )
    exact_target_windows = bool(split.get("exact_target_windows", False))
    split_windows = {
        "train": (split["train_start"], split["train_end"]),
        "validation": (split["validation_start"], split["validation_end"]),
        "test": (split["test_start"], split["test_end"]),
    }
    for split_name, mask in selected_splits:
        if exact_target_windows:
            predictions, samples, representations = _predict_target_window(
                fitted,
                snapshots,
                *split_windows[split_name],
                draws=int(config["sampling"]["draws"]),
                seed=int(config["sampling"]["seed"]),
                export_representations=export_representations,
                require_equal_samples=split_name != "train",
            )
        else:
            predictions, samples, representations = predict(
                fitted,
                snapshots,
                mask,
                draws=int(config["sampling"]["draws"]),
                seed=int(config["sampling"]["seed"]),
                export_representations=export_representations,
            )
        if export_representations:
            representations.insert(0, "split", split_name)
            representation_frames.append(representations)
        if split_name == "test":
            predictions.to_csv(output_dir / "predictions.csv", index=False)
            np.savez_compressed(output_dir / "predictive_samples.npz", samples=samples)
            score_predictions(predictions, samples).to_csv(
                output_dir / "metrics.csv", index=False
            )
        elif split_name == "validation" and export_validation_predictions:
            predictions.to_csv(
                output_dir / "validation_predictions.csv", index=False
            )
            np.savez_compressed(
                output_dir / "validation_predictive_samples.npz", samples=samples
            )
    if export_representations:
        pd.concat(representation_frames, ignore_index=True).to_csv(
            output_dir / "representations.csv", index=False
        )
    manifest = {
        "model": result_name,
        "architecture": model_name.upper(),
        "evaluation_horizons": config["forecast"]["horizons"],
        "units": len(snapshots.units),
        "candidate_edges_directed": int(graph.mask.sum()),
        "likelihood": fitted.likelihood,
        "outbreak_loss_weight": fitted.outbreak_loss_weight,
        "outbreak_positive_weight": fitted.outbreak_positive_weight,
        "outbreak_focal_gamma": fitted.outbreak_focal_gamma,
        "alert_probability_source": fitted.alert_probability_source,
        "test_origins": int(test.sum()),
        "refit_on_train_validation": bool(
            config["training"].get("refit_on_train_validation", False)
        ),
        "split": split,
        "monthly_data": _portable_manifest_path(resolve(root, data_config["monthly_path"])),
        "outbreak_thresholds": (
            _portable_manifest_path(resolve(root, threshold_path)) if threshold_path else None
        ),
        "outbreak_threshold_column": (
            str(data_config.get("outbreak_threshold_column", "epi_threshold"))
            if threshold_path
            else None
        ),
        "future_covariates": _portable_manifest_path(resolve(root, future_path)) if future_path else None,
        "future_covariate_columns": future_columns,
        "history_covariate_columns": list(data_config.get("history_covariate_columns", [])),
        "note": "H1, H2, and H3 are reported as separate evaluation endpoints.",
        "excluded_units": sorted(excluded_units),
        "excluded_unit_count": len(excluded_units),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output_dir
