#!/usr/bin/env python3
"""Leave-one-parent-area-out evaluation with strict training-node masking.

Each first-level administrative area is excluded from parameter fitting, loss
calculation, scale estimation, outbreak weighting, and messages into training
nodes. The held-out communes are then scored for the fixed 2021--2025 target
window. Together, the folds provide out-of-area predictions for every commune.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from dengue_h3.config import load_config
from dengue_h3.evaluation import score_predictions
from dengue_h3.pipeline import run_experiment


ROOT = Path(__file__).resolve().parents[2]
CLIMATE_COLUMNS = [
    "t2m_avg",
    "rh_avg",
    "tp_accum",
    "soil_moisture",
    "solar_downward",
    "dewpoint_c",
    "et0_mm",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "code/config/eval_2021_2025.yaml"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "code/data/processed/feature_suite/monthly_commune_open.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/pending/spatial_holdout"
    )
    parser.add_argument("--areas", nargs="+", help="Optional l1_code subset")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=9411)
    args = parser.parse_args()

    config_root, base_config = load_config(args.config)
    panel = pd.read_csv(
        args.data,
        dtype={"fcode": str, "l1_code": str},
        usecols=["fcode", "l1_code"],
    ).drop_duplicates()
    if panel["fcode"].duplicated().any():
        raise ValueError("A commune maps to more than one l1_code")
    panel = panel.sort_values("fcode").reset_index(drop=True)
    all_areas = sorted(panel["l1_code"].dropna().unique())
    areas = args.areas or all_areas
    unknown = sorted(set(areas).difference(all_areas))
    if unknown:
        raise ValueError(f"Unknown l1_code values: {unknown}")

    args.output.mkdir(parents=True, exist_ok=True)
    folds = {
        area: panel.loc[panel["l1_code"] == area, "fcode"].tolist() for area in areas
    }
    (args.output / "folds.json").write_text(
        json.dumps(folds, indent=2), encoding="utf-8"
    )

    all_units = tuple(panel["fcode"])
    unit_index = {unit: index for index, unit in enumerate(all_units)}
    prediction_frames: list[pd.DataFrame] = []
    fold_metric_frames: list[pd.DataFrame] = []
    full_samples: np.ndarray | None = None
    filled = np.zeros(len(all_units), dtype=bool)

    for area in areas:
        held_out = folds[area]
        label = f"L1-{area}"
        run_dir = args.output / label
        required = [run_dir / "predictions.csv", run_dir / "predictive_samples.npz"]
        if all(path.exists() for path in required):
            print(f"SKIP completed {label}", flush=True)
        else:
            config = deepcopy(base_config)
            config["data"]["monthly_path"] = str(args.data.resolve())
            config["data"]["climate_columns"] = CLIMATE_COLUMNS
            config["training"]["excluded_units"] = held_out
            config["training"]["seed"] = args.seed
            config["training"]["device"] = args.device
            config["sampling"]["draws"] = args.draws
            config["model"].update(
                {
                    "outbreak_loss_weight": 0.0,
                    "outbreak_positive_weight": 1.0,
                    "outbreak_focal_gamma": 0.0,
                    "alert_probability_source": "samples",
                }
            )
            print(
                f"RUN {label}: hold out {len(held_out)} communes on {args.device}",
                flush=True,
            )
            run_experiment(
                config_root,
                config,
                "M5",
                label,
                results_root=args.output,
                export_representations=False,
                export_validation_predictions=False,
            )

        predictions = pd.read_csv(
            run_dir / "predictions.csv", dtype={"fcode": str}
        )
        sample_cube = np.load(run_dir / "predictive_samples.npz")["samples"]
        if sample_cube.shape[1] != len(all_units):
            raise ValueError(
                f"Unexpected commune dimension in {label}: {sample_cube.shape}"
            )
        node_indices = np.asarray([unit_index[unit] for unit in held_out])
        held_predictions = predictions[predictions["fcode"].isin(held_out)].copy()
        held_samples = sample_cube[:, node_indices, :, :]
        expected_rows = (
            held_samples.shape[0] * held_samples.shape[1] * held_samples.shape[2]
        )
        if len(held_predictions) != expected_rows:
            raise ValueError(f"Prediction/sample key mismatch in {label}")
        held_predictions["model"] = "SPATIAL-HOLDOUT-M5"
        held_predictions.insert(0, "heldout_l1_code", area)
        held_metrics = score_predictions(
            held_predictions.drop(columns="heldout_l1_code"), held_samples
        )
        held_metrics.insert(0, "heldout_l1_code", area)
        held_predictions.to_csv(run_dir / "heldout_predictions.csv", index=False)
        held_metrics.to_csv(run_dir / "heldout_metrics.csv", index=False)
        prediction_frames.append(held_predictions)
        fold_metric_frames.append(held_metrics)

        if full_samples is None:
            full_samples = np.empty(
                (
                    sample_cube.shape[0],
                    len(all_units),
                    sample_cube.shape[2],
                    sample_cube.shape[3],
                ),
                dtype=sample_cube.dtype,
            )
        full_samples[:, node_indices, :, :] = held_samples
        filled[node_indices] = True

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics = pd.concat(fold_metric_frames, ignore_index=True)
    all_predictions.to_csv(
        args.output / "spatial_holdout_predictions.csv", index=False
    )
    fold_metrics.to_csv(args.output / "spatial_holdout_by_fold.csv", index=False)

    # A complete run (all areas) yields an exact cross-validated metric over all
    # 554 communes. A selected-area run writes fold-wise diagnostics only.
    if set(areas) == set(all_areas):
        if full_samples is None or not filled.all():
            raise RuntimeError("Not every commune received an out-of-area prediction")
        ordered = all_predictions.sort_values(
            ["issue_date", "fcode", "horizon"]
        ).drop(columns="heldout_l1_code")
        summary = score_predictions(ordered.reset_index(drop=True), full_samples)
        summary.to_csv(args.output / "spatial_holdout_summary.csv", index=False)
        shutil.copy2(
            args.output / "spatial_holdout_summary.csv",
            args.output.parent / "spatial_holdout_summary.csv",
        )

    provenance = {
        "design": "leave-one-l1_code-out",
        "areas": areas,
        "all_areas": all_areas,
        "communes": len(all_units),
        "data": str(args.data.resolve().relative_to(ROOT)),
        "data_sha256": sha256(args.data),
        "config": str(args.config.resolve().relative_to(ROOT)),
        "config_sha256": sha256(args.config),
        "model": "M5",
        "seed": args.seed,
        "draws": args.draws,
        "excluded_from": [
            "feature scaling",
            "training and validation loss",
            "outbreak class weighting",
            "messages into training nodes",
            "commune-specific embedding estimation",
        ],
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
