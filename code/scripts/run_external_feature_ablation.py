#!/usr/bin/env python3
"""Run each available external feature group as a separate GNN ablation."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from dengue_h3.config import load_config
from dengue_h3.pipeline import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--monthly", type=Path, required=True)
    parser.add_argument("--groups", type=Path, default=Path("config/external_feature_groups.yaml"))
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--model", default="GNN")
    args = parser.parse_args()

    root, base = load_config(args.config)
    groups_path = args.groups if args.groups.is_absolute() else root / args.groups
    spec = yaml.safe_load(groups_path.read_text(encoding="utf-8"))["groups"]
    available = set(pd.read_csv(args.monthly, nrows=0).columns)
    ran = []
    for name, group in spec.items():
        columns = list(group["columns"])
        missing = [column for column in columns if column not in available]
        if missing:
            print(f"SKIP {name}: missing {missing}")
            continue
        config = deepcopy(base)
        config["data"]["monthly_path"] = str(args.monthly.resolve())
        role = str(group["role"])
        if role == "history":
            config["data"]["history_covariate_columns"] = columns
        elif role == "climate":
            config["data"]["climate_columns"] = [*base["data"]["climate_columns"], *columns]
        else:
            raise ValueError(f"Unsupported role {role!r} for {name}")
        label = f"EXT-{name.upper().replace('_', '-')}"
        run_experiment(root, config, args.model, label, results_root=args.results_root)
        ran.append(label)
    print("ran:", ", ".join(ran) if ran else "none")


if __name__ == "__main__":
    main()
