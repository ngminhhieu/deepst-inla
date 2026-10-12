#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("horizon_dirs", nargs="+", type=Path)
    args = parser.parse_args()

    predictions, metrics, diagnostics = [], [], []
    args.output.mkdir(parents=True, exist_ok=True)
    seen = set()
    for directory in args.horizon_dirs:
        prediction = pd.read_csv(directory / "predictions.csv", dtype={"fcode": str})
        horizon_values = set(prediction["horizon"].astype(int).unique())
        if len(horizon_values) != 1:
            raise ValueError(f"Expected one horizon in {directory}: {horizon_values}")
        horizon = horizon_values.pop()
        if horizon in seen:
            raise ValueError(f"Duplicate horizon {horizon}")
        seen.add(horizon)
        predictions.append(prediction)
        metrics.append(pd.read_csv(directory / "metrics.csv"))
        diagnostics.append(pd.read_csv(directory / "diagnostics.csv"))
        for source_name in (
            f"samples_H{horizon}.bin",
            f"samples_H{horizon}_shape.csv",
        ):
            source = directory / source_name
            if source.exists():
                shutil.copy2(source, args.output / source.name)
    if seen != {1, 2, 3}:
        raise ValueError(f"Expected horizons 1, 2, 3; found {sorted(seen)}")
    pd.concat(predictions, ignore_index=True).sort_values(
        ["horizon", "issue_date", "fcode"]
    ).to_csv(args.output / "predictions.csv", index=False)
    pd.concat(metrics, ignore_index=True).sort_values("horizon").to_csv(
        args.output / "metrics.csv", index=False
    )
    pd.concat(diagnostics, ignore_index=True).sort_values("horizon").to_csv(
        args.output / "diagnostics.csv", index=False
    )


if __name__ == "__main__":
    main()
