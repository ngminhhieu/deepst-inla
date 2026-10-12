#!/usr/bin/env python3
"""Run every neural experiment across independent seeds and summarize uncertainty.

The standard deviation reported here is across independent training seeds.  The
predictive sampling seed is held fixed so that it does not inflate training-run
uncertainty with avoidable Monte Carlo noise.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

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
BASE_MODEL = {
    "outbreak_loss_weight": 0.0,
    "outbreak_positive_weight": 1.0,
    "outbreak_focal_gamma": 0.0,
    "alert_probability_source": "samples",
}
ALERT_MODEL = {
    "outbreak_loss_weight": 0.25,
    "outbreak_positive_weight": "auto",
    "outbreak_focal_gamma": 2.0,
    "alert_probability_source": "auxiliary",
}
EXPERIMENTS = {
    "CNN": {"architecture": "CNN"},
    "LSTM": {"architecture": "LSTM"},
    "CNN-LSTM": {"architecture": "CNN-LSTM"},
    "GNN": {"architecture": "GNN"},
    "LSTM-FULL": {"architecture": "LSTM-FULL"},
    "LSTM-GNN": {"architecture": "LSTM-GNN"},
    "CNN-LSTM-FULL": {"architecture": "CNN-LSTM-FULL"},
    "CNN-LSTM-GNN": {"architecture": "CNN-LSTM-GNN"},
    "M1": {"architecture": "M1"},
    "M2": {"architecture": "M2"},
    "M3": {"architecture": "M3"},
    "M4": {"architecture": "M4"},
    "M5": {"architecture": "M5"},
    "M6": {"architecture": "M6"},
    "TCN-H64": {"architecture": "M1", "model": {"hidden_size": 64}},
    "TCN-L24": {
        "architecture": "M1",
        "forecast": {"sequence_length": 24},
        "model": {"dilations": [1, 2, 4, 8]},
    },
    "M5-ALERT": {"architecture": "M5", "model": ALERT_MODEL},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure(base: dict, spec: dict, seed: int, device: str, draws: int, monthly: Path) -> dict:
    config = deepcopy(base)
    config["data"]["monthly_path"] = str(monthly.resolve())
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
    config["training"]["device"] = device
    config["sampling"]["draws"] = draws
    config["model"].update(BASE_MODEL)
    for section in ("forecast", "model"):
        config[section].update(spec.get(section, {}))
    return config


def collect(output: Path, experiments: list[str], seeds: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for experiment in experiments:
        for seed in seeds:
            path = output / "runs" / experiment / f"seed_{seed}" / "metrics.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame.insert(0, "seed", seed)
            frame.insert(0, "experiment", experiment)
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=("experiment", "seed", "horizon", *METRICS))
    return pd.concat(frames, ignore_index=True)


def summarize(all_metrics: pd.DataFrame, output: Path) -> None:
    all_metrics.to_csv(output / "all_metrics.csv", index=False)
    if all_metrics.empty:
        return
    seed_horizon = all_metrics.groupby(
        ["experiment", "seed", "horizon"], as_index=False
    )[list(METRICS)].mean()
    by_horizon = (
        seed_horizon.groupby(["experiment", "horizon"])[list(METRICS)]
        .agg(["mean", "std", "count"])
        .stack(0)
        .reset_index()
        .rename(columns={"level_2": "metric", "count": "seeds"})
    )
    by_horizon["scope"] = by_horizon["horizon"].map(lambda value: f"H{int(value)}")
    by_horizon = by_horizon[
        ["experiment", "scope", "horizon", "metric", "mean", "std", "seeds"]
    ]
    macro_seed = seed_horizon.groupby(["experiment", "seed"], as_index=False)[
        list(METRICS)
    ].mean()
    macro = (
        macro_seed.groupby("experiment")[list(METRICS)]
        .agg(["mean", "std", "count"])
        .stack(0)
        .reset_index()
        .rename(columns={"level_1": "metric", "count": "seeds"})
    )
    macro.insert(1, "scope", "mean_H1_H3")
    macro.insert(2, "horizon", pd.NA)
    summary = pd.concat([by_horizon, macro[by_horizon.columns]], ignore_index=True)
    summary["mean_plus_minus_sd"] = summary.apply(
        lambda row: (
            f"{row['mean']:.3f} ± {row['std']:.3f}"
            if pd.notna(row["std"])
            else f"{row['mean']:.3f}"
        ),
        axis=1,
    )
    by_horizon.to_csv(output / "summary_by_horizon.csv", index=False)
    macro[by_horizon.columns].to_csv(output / "summary_mean_H1_H3.csv", index=False)
    summary.to_csv(output / "summary_all_scopes.csv", index=False)
    summary.pivot_table(
        index=["experiment", "scope"],
        columns="metric",
        values="mean_plus_minus_sd",
        aggfunc="first",
    ).reset_index().to_csv(output / "table_mean_plus_minus_sd.csv", index=False)


def write_status(
    output: Path,
    experiments: list[str],
    seeds: list[int],
    *,
    state: str,
    active: str | None,
    started_at: str,
) -> None:
    completed = []
    for experiment in experiments:
        for seed in seeds:
            run_dir = output / "runs" / experiment / f"seed_{seed}"
            if (run_dir / "COMPLETED").exists() and (run_dir / "metrics.csv").exists():
                completed.append(f"{experiment}/seed_{seed}")
    payload = {
        "state": state,
        "started_at": started_at,
        "updated_at": utc_now(),
        "active": active,
        "experiments": experiments,
        "seeds": seeds,
        "completed_runs": len(completed),
        "total_runs": len(experiments) * len(seeds),
        "completed": completed,
    }
    (output / "status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "code/config/eval_2021_2025.yaml"
    )
    parser.add_argument(
        "--monthly",
        type=Path,
        default=ROOT / "code/data/processed/feature_suite/monthly_commune_open.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/uncertainty_all_experiments"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[9411, 9412, 9413, 9414, 9415])
    parser.add_argument("--experiments", nargs="+", choices=sorted(EXPERIMENTS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument(
        "--keep-predictive-samples",
        action="store_true",
        help="Keep large predictive_samples.npz files after metrics are computed.",
    )
    args = parser.parse_args()
    experiments = args.experiments or list(EXPERIMENTS)
    if not args.monthly.exists():
        raise FileNotFoundError(args.monthly)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "runs").mkdir(exist_ok=True)
    config_root, base_config = load_config(args.config)
    started_at = utc_now()
    write_status(
        args.output, experiments, args.seeds, state="running", active=None, started_at=started_at
    )

    for experiment in experiments:
        spec = EXPERIMENTS[experiment]
        for seed in args.seeds:
            relative = Path("runs") / experiment / f"seed_{seed}"
            run_dir = args.output / relative
            if (run_dir / "COMPLETED").exists() and (run_dir / "metrics.csv").exists():
                print(f"SKIP completed {relative}", flush=True)
                continue
            active = str(relative)
            write_status(
                args.output,
                experiments,
                args.seeds,
                state="running",
                active=active,
                started_at=started_at,
            )
            config = configure(base_config, spec, seed, args.device, args.draws, args.monthly)
            print(f"RUN {active} architecture={spec['architecture']} device={args.device}", flush=True)
            run_experiment(
                config_root,
                config,
                str(spec["architecture"]),
                f"seed_{seed}",
                results_root=args.output / "runs" / experiment,
                export_representations=False,
                export_validation_predictions=False,
            )
            if not args.keep_predictive_samples:
                sample_path = run_dir / "predictive_samples.npz"
                if sample_path.exists():
                    sample_path.unlink()
            (run_dir / "COMPLETED").write_text(utc_now() + "\n", encoding="utf-8")
            summarize(collect(args.output, experiments, args.seeds), args.output)
            print(f"DONE {active}", flush=True)

    summarize(collect(args.output, experiments, args.seeds), args.output)
    write_status(
        args.output,
        experiments,
        args.seeds,
        state="complete",
        active=None,
        started_at=started_at,
    )
    print(args.output / "table_mean_plus_minus_sd.csv", flush=True)


if __name__ == "__main__":
    main()
