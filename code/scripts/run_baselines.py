#!/usr/bin/env python3
from __future__ import annotations

import argparse

from dengue_h3.baselines import run_baseline
from dengue_h3.config import load_config, resolve
from dengue_h3.data import (
    build_snapshots,
    load_monthly,
    load_outbreak_thresholds,
    split_mask,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/study_lognormal.yaml")
    args = parser.parse_args()
    root, config = load_config(args.config)
    monthly = load_monthly(
        resolve(root, config["data"]["monthly_path"]), config["data"]
    )
    threshold_path = config["data"].get("outbreak_thresholds_path")
    resolved_threshold_path = resolve(root, threshold_path) if threshold_path else None
    thresholds = (
        load_outbreak_thresholds(
            resolved_threshold_path,
            unit_column=config["data"].get("outbreak_threshold_unit_column", "fcode"),
            threshold_column=config["data"].get("outbreak_threshold_column", "epi_threshold"),
        )
        if resolved_threshold_path is not None and resolved_threshold_path.exists()
        else None
    )
    data = build_snapshots(monthly, config, outbreak_thresholds=thresholds)
    test = split_mask(
        data, config["split"]["test_start"], config["split"]["test_end"]
    )
    exact = bool(config["split"].get("exact_target_windows", False))
    for name in ("seasonal_naive", "province_allocation"):
        run_baseline(
            name,
            data,
            None if exact else test,
            root.parent / "results" / name,
            draws=int(config["sampling"]["draws"]),
            seed=int(config["sampling"]["seed"]),
            target_start=config["split"]["test_start"] if exact else None,
            target_end=config["split"]["test_end"] if exact else None,
        )


if __name__ == "__main__":
    main()
