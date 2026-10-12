#!/usr/bin/env python3
"""Five-seed validation reruns for every configuration displayed in Figure 7."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_h3.config import load_config
from dengue_h3.evaluation import score_predictions
from dengue_h3.pipeline import run_experiment


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (9411, 9412, 9413, 9414, 9415)
BASE = ("t2m_avg", "rh_avg", "tp_accum")
FULL = (*BASE, "soil_moisture", "solar_downward", "dewpoint_c", "et0_mm")
ARCHITECTURES = {
    "TCN-GNN": ("M5", FULL),
    "TCN-GNN + dynamic spatial gate": ("M6", FULL),
    "CNN-LSTM-GNN": ("CNN-LSTM-GNN", FULL),
    "LSTM-GNN": ("LSTM-GNN", FULL),
    "Last-state GNN": ("GNN", FULL),
}
# Full set is shared with the selected TCN-GNN architecture and is therefore
# not retrained a sixth time here. The summarizer imports those five runs.
FEATURES = {
    "Basic climate": ("M5", BASE),
    "Basic + ET0": ("M5", (*BASE, "et0_mm")),
    "Basic + soil moisture": ("M5", (*BASE, "soil_moisture")),
    "Basic + solar radiation": ("M5", (*BASE, "solar_downward")),
    "Basic + dew point": ("M5", (*BASE, "dewpoint_c")),
}
METRICS = ("crps", "mae", "outbreak_brier", "outbreak_sensitivity", "outbreak_f1")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: str) -> str:
    return (value.lower().replace(" + ", "-").replace(" ", "-")
            .replace("--", "-").replace("/", "-"))


def configured(base: dict, architecture: str, columns: tuple[str, ...], seed: int) -> dict:
    config = deepcopy(base)
    config["data"]["monthly_path"] = str(
        (ROOT / "code/data/processed/feature_suite/monthly_commune_open.csv").resolve()
    )
    config["data"]["climate_columns"] = list(columns)
    # Keep the audited MOH threshold table from the base configuration.
    config["training"]["seed"] = seed
    config["training"]["device"] = "cpu"
    # Validation predictions must come from the model trained only through 2018.
    config["training"]["refit_on_train_validation"] = False
    config["sampling"]["draws"] = 500
    return config


def status(root: Path, group: str, experiments: dict[str, tuple[str, tuple[str, ...]]]) -> None:
    complete = []
    for label in experiments:
        for seed in SEEDS:
            if (root / "runs" / group / slug(label) / f"seed_{seed}" / "COMPLETED").exists():
                complete.append(f"{label}/seed_{seed}")
    payload = {"group": group, "updated_at": now(), "completed_runs": len(complete),
               "total_runs": len(experiments) * len(SEEDS), "completed": complete}
    (root / f"status_{group}.json").write_text(json.dumps(payload, indent=2) + "\n")


def worker(group: str, root: Path) -> None:
    experiments = ARCHITECTURES if group == "architectures" else FEATURES
    config_root, base = load_config(ROOT / "code/config/eval_2021_2025.yaml")
    for label, (architecture, columns) in experiments.items():
        for seed in SEEDS:
            run_dir = root / "runs" / group / slug(label) / f"seed_{seed}"
            if (run_dir / "COMPLETED").exists() and (run_dir / "validation_metrics.csv").exists():
                print(f"SKIP {group} {label} seed={seed}", flush=True)
                continue
            print(f"RUN {group} {label} seed={seed} architecture={architecture}", flush=True)
            config = configured(base, architecture, columns, seed)
            output = run_experiment(
                config_root, config, architecture, f"seed_{seed}",
                results_root=root / "runs" / group / slug(label),
                export_representations=False, export_validation_predictions=True,
            )
            predictions = pd.read_csv(output / "validation_predictions.csv")
            with np.load(output / "validation_predictive_samples.npz") as archive:
                samples = archive["samples"]
            score_predictions(predictions, samples, split="validation").to_csv(
                output / "validation_metrics.csv", index=False
            )
            for name in ("model.pt", "predictions.csv", "predictive_samples.npz",
                         "validation_predictions.csv", "validation_predictive_samples.npz"):
                path = output / name
                if path.exists():
                    path.unlink()
            (output / "COMPLETED").write_text(now() + "\n")
            status(root, group, experiments)
            print(f"DONE {group} {label} seed={seed}", flush=True)
    (root / f"{group.upper()}_COMPLETE").write_text(now() + "\n")


def summarize(root: Path) -> None:
    if not (root / "ARCHITECTURES_COMPLETE").exists() or not (root / "FEATURES_COMPLETE").exists():
        raise RuntimeError("Both Figure 7 workers must finish before summarization")
    rows = []
    groups = (("architecture", "architectures", ARCHITECTURES),
              ("feature", "features", FEATURES))
    for kind, directory, experiments in groups:
        for label in experiments:
            for seed in SEEDS:
                path = root / "runs" / directory / slug(label) / f"seed_{seed}" / "validation_metrics.csv"
                frame = pd.read_csv(path)
                frame.insert(0, "seed", seed)
                frame.insert(0, "configuration", label)
                frame.insert(0, "ablation", kind)
                rows.append(frame)
    # Reuse the identical selected TCN-GNN/full-set fits instead of retraining.
    selected = [frame.copy() for frame in rows if
                frame["ablation"].iat[0] == "architecture" and
                frame["configuration"].iat[0] == "TCN-GNN"]
    for frame in selected:
        frame["ablation"] = "feature"
        frame["configuration"] = "Full set"
        rows.append(frame)
    raw = pd.concat(rows, ignore_index=True)
    raw.to_csv(root / "validation_metrics_all_seeds.csv", index=False)
    seed_macro = raw.groupby(["ablation", "configuration", "seed"], as_index=False)[list(METRICS)].mean()
    long = seed_macro.melt(id_vars=["ablation", "configuration", "seed"],
                           value_vars=METRICS, var_name="metric", value_name="value")
    summary = (long.groupby(["ablation", "configuration", "metric"])["value"]
               .agg(mean="mean", sd="std", seeds="count").reset_index())
    summary["mean_plus_minus_sd"] = summary.apply(
        lambda r: f"{r['mean']:.3f} ± {r['sd']:.3f}", axis=1
    )
    summary.to_csv(root / "figure7_mean_sd.csv", index=False)
    summary.pivot(index=["ablation", "configuration"], columns="metric",
                  values="mean_plus_minus_sd").reset_index().to_csv(
        root / "figure7_table_mean_plus_minus_sd.csv", index=False
    )
    (root / "COMPLETE").write_text(now() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("architectures", "features", "summarize"))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/figure7_multiseed_validation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "summarize":
        summarize(args.output)
    else:
        worker(args.mode, args.output)


if __name__ == "__main__":
    main()
