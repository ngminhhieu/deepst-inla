#!/usr/bin/env python3
"""Phase B1: rescore 17x5 saved neural checkpoints with audited MOH labels.

The original predictive-draw files were not retained. This script performs
inference only (no fitting) from the preserved checkpoints and recreates the
500 draws with the original Monte Carlo seed.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from dengue_h3.config import load_config, resolve  # noqa: E402
from dengue_h3.data import build_snapshots, load_monthly, load_outbreak_thresholds  # noqa: E402
from dengue_h3.distributions import predictive_sample  # noqa: E402
from dengue_h3.evaluation import outbreak_classification_metrics  # noqa: E402
from dengue_h3.graph import build_candidate_graph  # noqa: E402
from dengue_h3.models import DengueForecastNet  # noqa: E402
from dengue_h3.specs import get_spec  # noqa: E402
from dengue_h3.training import _apply_scale  # noqa: E402
from run_all_multiseed_uncertainty import EXPERIMENTS, configure  # noqa: E402

CONFIG = ROOT / "code/config/eval_2021_2025.yaml"
MONTHLY = ROOT / "code/data/processed/feature_suite/monthly_commune_open.csv"
CHECKPOINTS = ROOT / "artifacts/model_weights/uncertainty_all_experiments/runs"
ORIGINAL = ROOT / "results/uncertainty_all_experiments/all_metrics.csv"
OUT = ROOT / "results/rev_20261012/seed_suite_audited"
SEEDS = (9411, 9412, 9413, 9414, 9415)
DRAW_SEED = 1234
DRAW_COUNT = 500
OUTBREAK_COLUMNS = [
    "outbreak_brier", "outbreak_decision_threshold", "outbreak_observed_positive",
    "outbreak_predicted_positive", "outbreak_tp", "outbreak_fp", "outbreak_fn",
    "outbreak_tn", "outbreak_accuracy", "outbreak_precision", "outbreak_specificity",
    "outbreak_recall", "outbreak_sensitivity", "outbreak_true_negative_rate",
    "outbreak_balanced_accuracy", "outbreak_f1",
]


def load_checkpoint(path):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    return torch.load(path, map_location="cpu", weights_only=False)


def build_context(base, spec, seed, cache):
    config = configure(base, spec, seed, "cpu", DRAW_COUNT, MONTHLY)
    key = int(config["forecast"]["sequence_length"])
    if key not in cache:
        data_cfg = config["data"]
        monthly = load_monthly(MONTHLY, data_cfg)
        thresholds = load_outbreak_thresholds(
            resolve(ROOT / "code", data_cfg["outbreak_thresholds_path"]),
            unit_column=data_cfg.get("outbreak_threshold_unit_column", "fcode"),
            threshold_column=data_cfg["outbreak_threshold_column"],
        )
        snapshots = build_snapshots(monthly, config, outbreak_thresholds=thresholds)
        graph_cfg = config["graph"]
        graph = build_candidate_graph(
            snapshots.units, resolve(ROOT / "code", data_cfg["centroids_path"]),
            resolve(ROOT / "code", data_cfg["adjacency_path"]),
            k_nearest=int(graph_cfg["k_nearest"]),
            maximum_distance_km=float(graph_cfg["maximum_distance_km"]),
            include_administrative_edges=bool(graph_cfg["include_administrative_edges"]),
        )
        cache[key] = snapshots, graph
    return config, cache[key][0], cache[key][1]


def network_from_checkpoint(architecture, config, snapshots, graph, saved):
    model_cfg = config["model"]
    network = DengueForecastNet(
        spec=get_spec(architecture), local_feature_size=snapshots.local.shape[-1],
        climate_feature_size=snapshots.climate.shape[-1],
        seasonal_feature_size=snapshots.seasonal.shape[-1], node_count=len(snapshots.units),
        horizon_count=len(snapshots.horizons), hidden_size=int(model_cfg["hidden_size"]),
        commune_embedding_size=int(model_cfg["commune_embedding_size"]),
        dilations=tuple(int(x) for x in model_cfg["dilations"]),
        kernel_size=int(model_cfg["kernel_size"]), dropout=float(model_cfg["dropout"]),
        minimum_dispersion=float(model_cfg["minimum_dispersion"]),
        likelihood=str(saved["likelihood"]), maximum_log_sigma=float(model_cfg["maximum_log_sigma"]),
        candidate_mask=torch.from_numpy(graph.mask), edge_features=torch.from_numpy(graph.edge_features),
        fixed_weights=torch.from_numpy(graph.fixed_weights), graph_heads=int(model_cfg["graph_heads"]),
    )
    network.load_state_dict(saved["state_dict"])
    return network.eval()


def score_run(experiment, seed, base, cache, original):
    spec = EXPERIMENTS[experiment]
    config, snapshots, graph = build_context(base, spec, seed, cache)
    saved = load_checkpoint(CHECKPOINTS / experiment / ("seed_%d" % seed) / "model.pt")
    network = network_from_checkpoint(spec["architecture"], config, snapshots, graph, saved)
    target_dates = pd.to_datetime(snapshots.target_dates.reshape(-1)).to_numpy().reshape(
        snapshots.target_dates.shape)
    lower, upper = np.datetime64("2021-01-01"), np.datetime64("2025-12-01")
    selected = ((target_dates >= lower) & (target_dates <= upper)).any(axis=1)
    with torch.no_grad():
        location, scale, _, _, logits = network(
            torch.from_numpy(_apply_scale(snapshots.local[selected], saved["local_scale"])),
            torch.from_numpy(_apply_scale(snapshots.climate[selected], saved["climate_scale"])),
            torch.from_numpy(_apply_scale(snapshots.seasonal[selected], saved["seasonal_scale"])),
            return_outbreak_logits=True,
        )
    parameter_1, parameter_2 = location.numpy(), scale.numpy()
    auxiliary = torch.sigmoid(logits).numpy()
    samples = None
    probability_source = str(saved.get("alert_probability_source", "samples"))
    if probability_source != "auxiliary":
        samples = predictive_sample(parameter_1, parameter_2, likelihood=str(saved["likelihood"]),
                                    draws=DRAW_COUNT, rng=np.random.default_rng(DRAW_SEED))
    rows = []
    selected_dates = target_dates[selected]
    for h_index, horizon in enumerate(snapshots.horizons):
        keep = (selected_dates[:, h_index] >= lower) & (selected_dates[:, h_index] <= upper)
        observed = snapshots.targets[selected][:, :, h_index][keep].reshape(-1)
        threshold = snapshots.outbreak_thresholds[selected][:, :, h_index][keep].reshape(-1)
        if probability_source == "auxiliary":
            probability = auxiliary[:, :, h_index][keep].reshape(-1)
        else:
            probability = (samples[:, :, h_index, :][keep] >
                           threshold.reshape(keep.sum(), len(snapshots.units), 1)).mean(axis=-1).reshape(-1)
        metrics = outbreak_classification_metrics(observed, threshold, probability, 0.5)
        old = original.query("experiment == @experiment and seed == @seed and horizon == @horizon").iloc[0].copy()
        old["outbreak_brier"] = float(np.mean((probability - (observed > threshold)) ** 2))
        for name, value in metrics.items():
            old[name] = value
        rows.append(old)
    del samples, network
    return pd.DataFrame(rows)


def summaries(all_metrics):
    metrics = ["mae", "rmse", "crps", "outbreak_brier", "outbreak_precision",
               "outbreak_specificity", "outbreak_sensitivity", "outbreak_f1"]
    seed_h = all_metrics.groupby(["experiment", "seed", "horizon"], as_index=False)[metrics].mean()
    macro_seed = seed_h.groupby(["experiment", "seed"], as_index=False)[metrics].mean()
    macro = macro_seed.groupby("experiment")[metrics].agg(["mean", "std", "count"]).stack(0).reset_index()
    macro = macro.rename(columns={"level_1": "metric", "count": "seeds"})
    macro.insert(1, "scope", "mean_H1_H3"); macro.insert(2, "horizon", pd.NA)
    return macro[["experiment", "scope", "horizon", "metric", "mean", "std", "seeds"]]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(frame, path, inputs):
    frame.to_csv(path, index=False)
    provenance = {"inputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in inputs],
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "command": " ".join(sys.argv), "seeds": list(SEEDS) + [DRAW_SEED],
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(), "network_refit": False,
                  "note": "500 predictive draws regenerated from preserved checkpoints because original draws were not retained"}
    path.with_suffix(path.suffix + ".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, base = load_config(CONFIG)
    original = pd.read_csv(ORIGINAL)
    progress_path = OUT / "rescored_all_metrics.csv"
    completed = pd.read_csv(progress_path) if progress_path.exists() else pd.DataFrame()
    if not completed.empty:
        counts = completed.groupby(["experiment", "seed"])["horizon"].transform("nunique")
        completed = completed[counts.eq(3)].copy()
    cache = {}
    for experiment in EXPERIMENTS:
        for seed in SEEDS:
            if not completed.empty and ((completed.experiment == experiment) &
                                        (completed.seed == seed)).sum() == 3:
                print("SKIP", experiment, seed, flush=True); continue
            print("RUN", experiment, seed, flush=True)
            frame = score_run(experiment, seed, base, cache, original)
            completed = pd.concat([completed, frame], ignore_index=True)
            write(completed, progress_path, [CONFIG, MONTHLY, ORIGINAL])
            print("DONE", experiment, seed, flush=True)
    summary = summaries(completed)
    write(summary, OUT / "summary_mean_H1_H3.csv", [progress_path])
    print(summary.query("experiment == 'M5'").to_string(index=False))


if __name__ == "__main__":
    main()
