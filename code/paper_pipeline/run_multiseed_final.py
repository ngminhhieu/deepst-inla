#!/usr/bin/env python3
"""Run and summarize independent seeds for the principal neural paths."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil

import pandas as pd

from dengue_h3.config import load_config
from dengue_h3.pipeline import run_experiment


ROOT = Path(__file__).resolve().parents[2]
METRICS = (
    "mae",
    "rmse",
    "crps",
    "outbreak_brier",
    "outbreak_precision",
    "outbreak_specificity",
    "outbreak_sensitivity",
    "outbreak_f1",
)
FAMILY_CONFIG = {
    "base": {
        "outbreak_loss_weight": 0.0,
        "outbreak_positive_weight": 1.0,
        "outbreak_focal_gamma": 0.0,
        "alert_probability_source": "samples",
    },
    "alert": {
        "outbreak_loss_weight": 0.25,
        "outbreak_positive_weight": "auto",
        "outbreak_focal_gamma": 2.0,
        "alert_probability_source": "auxiliary",
    },
}


def summarize(all_metrics: pd.DataFrame, output: Path) -> None:
    seed_horizon = (
        all_metrics.groupby(["family", "seed", "horizon"], as_index=False)[list(METRICS)]
        .mean()
    )
    horizon = (
        seed_horizon.groupby(["family", "horizon"])[list(METRICS)]
        .agg(["mean", "std"])
        .stack(0)
        .reset_index()
        .rename(columns={"level_2": "metric"})
    )
    horizon["scope"] = horizon["horizon"].map(lambda value: f"H{int(value)}")
    macro_by_seed = (
        seed_horizon.groupby(["family", "seed"], as_index=False)[list(METRICS)].mean()
    )
    macro = (
        macro_by_seed.groupby("family")[list(METRICS)]
        .agg(["mean", "std"])
        .stack(0)
        .reset_index()
        .rename(columns={"level_1": "metric"})
    )
    macro["horizon"] = pd.NA
    macro["scope"] = "mean_H1_H3"
    summary = pd.concat(
        [
            horizon[["family", "scope", "horizon", "metric", "mean", "std"]],
            macro[["family", "scope", "horizon", "metric", "mean", "std"]],
        ],
        ignore_index=True,
    )
    counts = all_metrics.groupby("family")["seed"].nunique().rename("seeds").reset_index()
    summary = summary.merge(counts, on="family", how="left")
    summary.to_csv(output / "multiseed_summary.csv", index=False)
    macro_by_seed.to_csv(output / "multiseed_macro_by_seed.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "code/config/eval_2021_2025.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[9411, 9412, 9413, 9414, 9415])
    parser.add_argument("--families", nargs="+", choices=sorted(FAMILY_CONFIG), default=["base", "alert"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument(
        "--monthly",
        type=Path,
        default=ROOT / "code/data/processed/feature_suite/monthly_commune_open.csv",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/pending/multiseed")
    args = parser.parse_args()

    config_root, base_config = load_config(args.config)
    if not args.monthly.exists():
        raise FileNotFoundError(args.monthly)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for family in args.families:
        for seed in args.seeds:
            label = f"{family.upper()}-SEED-{seed}"
            run_dir = args.output / label
            metrics_path = run_dir / "metrics.csv"
            if metrics_path.exists() and (run_dir / "predictions.csv").exists():
                print(f"SKIP completed {label}", flush=True)
            else:
                config = deepcopy(base_config)
                config["data"]["monthly_path"] = str(args.monthly.resolve())
                config["data"]["climate_columns"] = [
                    "t2m_avg",
                    "rh_avg",
                    "tp_accum",
                    "soil_moisture",
                    "solar_downward",
                    "dewpoint_c",
                    "et0_mm",
                ]
                config["training"]["seed"] = seed
                config["training"]["device"] = args.device
                config["sampling"]["draws"] = args.draws
                config["model"].update(FAMILY_CONFIG[family])
                print(f"RUN {label} on {args.device}", flush=True)
                run_experiment(
                    config_root,
                    config,
                    "M5",
                    label,
                    results_root=args.output,
                    export_representations=False,
                    export_validation_predictions=True,
                )
            metrics = pd.read_csv(metrics_path)
            metrics.insert(0, "seed", seed)
            metrics.insert(0, "family", family)
            rows.append(metrics)
            all_metrics = pd.concat(rows, ignore_index=True)
            all_metrics.to_csv(args.output / "multiseed_all.csv", index=False)
            summarize(all_metrics, args.output)
            shutil.copy2(
                args.output / "multiseed_summary.csv",
                args.output.parent / "multiseed_summary.csv",
            )
    print(args.output / "multiseed_summary.csv")


if __name__ == "__main__":
    main()
