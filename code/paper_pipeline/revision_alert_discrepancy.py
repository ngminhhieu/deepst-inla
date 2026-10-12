#!/usr/bin/env python3
"""Phase C: identify which probability path explains DeepST-Alert results."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOCKED = ROOT / "artifacts/recovered_from_trash/results/model_runs_train2010_2018_valid2019_2020_test2021_2025_external_thresholds/DEEPST-ALERT"
SEED = ROOT / "results/pending/multiseed/ALERT-SEED-9411"
MOH = ROOT / "code/data/raw/operational_2026/commune_monthly_thresholds.csv"
OUT = ROOT / "results/rev_20261012/alert_discrepancy"


def metrics(frame, probability_column):
    rows = []
    for horizon, part in frame.groupby("horizon"):
        y = part.target_cases.to_numpy(float) > part.epi_threshold.to_numpy(float)
        p = part[probability_column].to_numpy(float)
        a = p >= 0.5
        tp, fp = np.sum(y & a), np.sum(~y & a)
        fn, tn = np.sum(y & ~a), np.sum(~y & ~a)
        ratio = lambda x, z: float(x / z) if z else 0.0
        sensitivity, specificity, precision = ratio(tp, tp + fn), ratio(tn, tn + fp), ratio(tp, tp + fp)
        rows.append({"horizon": int(horizon), "brier": np.mean((p - y) ** 2),
                     "sensitivity": sensitivity, "specificity": specificity,
                     "precision": precision, "f1": ratio(2 * precision * sensitivity, precision + sensitivity)})
    return pd.DataFrame(rows)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    moh = pd.read_csv(MOH, dtype={"fcode": str})[["fcode", "target_date", "epi_threshold"]]
    moh["target_date"] = pd.to_datetime(moh.target_date)
    outputs = []
    paths = []
    for pipeline, directory, sources in [
        ("locked_seed_9411", LOCKED, ["outbreak_probability_samples", "outbreak_probability_auxiliary"]),
        ("five_seed_pipeline_seed_9411", SEED, ["outbreak_probability_auxiliary"]),
    ]:
        path = directory / "predictions.csv"; paths.append(path)
        frame = pd.read_csv(path, dtype={"fcode": str}); frame["target_date"] = pd.to_datetime(frame.target_date)
        frame = frame.drop(columns=["epi_threshold"], errors="ignore").merge(
            moh, on=["fcode", "target_date"], validate="many_to_one")
        if pipeline == "locked_seed_9411":
            sample_path = directory / "predictive_samples.npz"; paths.append(sample_path)
            cube = np.load(sample_path)["samples"]
            for index, horizon in enumerate(sorted(frame.horizon.unique())):
                mask = frame.horizon.eq(horizon)
                matrix = cube[:, :, index, :].reshape(-1, cube.shape[-1])
                frame.loc[mask, "outbreak_probability_samples"] = (
                    matrix > frame.loc[mask, "epi_threshold"].to_numpy(float)[:, None]
                ).mean(axis=1)
        manifest_path = directory / "manifest.json"; paths.append(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        for source in sources:
            by_horizon = metrics(frame, source)
            row = {"pipeline": pipeline, "seed": 9411, "probability_source_scored": source,
                   "declared_probability_source": manifest.get("alert_probability_source"),
                   "outbreak_loss_weight": manifest.get("outbreak_loss_weight"),
                   "outbreak_focal_gamma": manifest.get("outbreak_focal_gamma"),
                   "outbreak_positive_weight": json.dumps(manifest.get("outbreak_positive_weight")),
                   "refit_on_train_validation": manifest.get("refit_on_train_validation", pipeline.startswith("five_seed"))}
            row.update({name: by_horizon[name].mean() for name in
                        ["brier", "sensitivity", "specificity", "precision", "f1"]})
            outputs.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "seed_9411_probability_paths.csv"
    pd.DataFrame(outputs).to_csv(path, index=False)
    provenance = {"inputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in [*paths, MOH]],
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "command": " ".join(sys.argv), "seeds": [9411],
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(), "refit": False,
                  "label": "audited MOH", "cutoff": 0.5}
    path.with_suffix(path.suffix + ".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(pd.DataFrame(outputs).to_string(index=False))


if __name__ == "__main__":
    main()
