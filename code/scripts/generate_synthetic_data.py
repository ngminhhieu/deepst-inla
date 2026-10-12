#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--communes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    units = [f"S{index:03d}" for index in range(args.communes)]
    dates = pd.date_range("2010-01-01", "2025-12-01", freq="MS")
    rows = []
    for unit_index, unit in enumerate(units):
        population = 10_000 + unit_index * 1_000
        province = f"P{unit_index // max(args.communes // 3, 1) + 1}"
        local_risk = rng.lognormal(0, 0.25)
        for date in dates:
            season = 1.2 + np.sin(2 * np.pi * (date.month - 4) / 12)
            temperature = 298 + 3 * np.sin(2 * np.pi * (date.month - 3) / 12)
            humidity = 72 + 8 * np.sin(2 * np.pi * (date.month - 5) / 12)
            rainfall = max(0, 0.15 + 0.12 * np.sin(2 * np.pi * (date.month - 5) / 12))
            mean_cases = max(0.1, local_risk * season * population / 20_000)
            rows.append(
                {
                    "fcode": unit,
                    "l1_code": province,
                    "year": date.year,
                    "month": date.month,
                    "obs_dengue_cases": rng.negative_binomial(2, 2 / (2 + mean_cases)),
                    "population": population,
                    "t2m_avg": temperature + rng.normal(0, 0.5),
                    "rh_avg": humidity + rng.normal(0, 1),
                    "tp_accum": max(0, rainfall + rng.normal(0, 0.02)),
                }
            )
    pd.DataFrame(rows).to_csv(output / "monthly_commune.csv", index=False)
    angles = np.linspace(0, 2 * np.pi, len(units), endpoint=False)
    pd.DataFrame(
        {
            "fcode": units,
            "longitude": 105.5 + 0.5 * np.cos(angles),
            "latitude": 10.2 + 0.5 * np.sin(angles),
        }
    ).to_csv(output / "commune_centroids.csv", index=False)
    edges = []
    for index, unit in enumerate(units):
        neighbour = units[(index + 1) % len(units)]
        edges.extend(
            [
                {"from_fcode": unit, "to_fcode": neighbour},
                {"from_fcode": neighbour, "to_fcode": unit},
            ]
        )
    pd.DataFrame(edges).drop_duplicates().to_csv(
        output / "administrative_adjacency.csv", index=False
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
