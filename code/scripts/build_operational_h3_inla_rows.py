#!/usr/bin/env python3
"""Build the 2026 H3 rows used by the established INLA design."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = pd.read_csv(args.panel, dtype={"fcode": str})
    panel["date"] = pd.to_datetime(dict(year=panel.year, month=panel.month, day=1))
    rows = []
    for fcode, frame in panel.groupby("fcode", sort=True):
        frame = frame.sort_values("date").reset_index(drop=True)
        lookup = {date: index for index, date in enumerate(frame.date)}
        cases = frame.obs_dengue_cases.to_numpy(float)
        for issue in pd.date_range("2026-01-01", "2026-08-01", freq="MS"):
            target = issue + pd.DateOffset(months=3)
            issue_index, target_index = lookup[issue], lookup[target]
            recent = issue_index - np.array([0, 1, 2, 5, 11])
            seasonal = target_index - 12 * np.arange(1, 6)
            seasonal_raw = cases[seasonal]
            seasonal_log = np.log1p(seasonal_raw)
            row = {
                "split": "test", "fcode": fcode,
                "issue_date": issue.date(), "target_date": target.date(),
                "horizon": 3, "target_cases": cases[target_index],
                "population": float(frame.loc[target_index, "population"]),
                "outbreak_threshold": float(seasonal_raw.mean() + 2 * seasonal_raw.std(ddof=1)),
                "log_recent_0": np.log1p(cases[recent[0]]),
                "log_recent_1": np.log1p(cases[recent[1]]),
                "log_recent_2": np.log1p(cases[recent[2]]),
                "log_recent_5": np.log1p(cases[recent[3]]),
                "log_recent_11": np.log1p(cases[recent[4]]),
                "seasonal_mean": seasonal_log.mean(),
                "seasonal_sd": seasonal_log.std(ddof=1),
                "seasonal_trend": seasonal_log[0] - seasonal_log[-1],
                "seasonal_zero_share": np.mean(seasonal_raw == 0),
                "target_sine": np.sin(2 * np.pi * target.month / 12),
                "target_cosine": np.cos(2 * np.pi * target.month / 12),
            }
            for lag in range(1, 6):
                row[f"seasonal_log_{lag}"] = seasonal_log[lag - 1]
                row[f"seasonal_available_{lag}"] = 1.0
            rows.append(row)
    output = pd.DataFrame(rows).sort_values(["target_date", "fcode"])
    if len(output) != 8 * panel.fcode.nunique():
        raise ValueError("Unexpected operational H3 row count")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"rows={len(output)} communes={output.fcode.nunique()}")


if __name__ == "__main__":
    main()
