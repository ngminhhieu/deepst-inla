#!/usr/bin/env python3
"""Apply the validation-locked DeepST-INLA H3 distribution policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exact_inla_deep_ensemble import load_exact_horizon


KEYS = ["fcode", "issue_date", "target_date", "horizon"]


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return f"external:{resolved.name}"


def load_component(directory: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(directory / "predictions_h3.csv", dtype={"fcode": str})
    samples = np.load(directory / "predictive_samples_h3.npz")["samples"]
    if len(frame) != len(samples):
        raise ValueError(f"Prediction/sample mismatch in {directory}")
    return frame, samples


def align(template: pd.DataFrame, frame: pd.DataFrame, samples: np.ndarray) -> np.ndarray:
    source = frame.copy()
    source["_row"] = np.arange(len(source))
    aligned = template[KEYS].merge(source[KEYS + ["_row"]], on=KEYS, how="left", validate="one_to_one")
    if aligned["_row"].isna().any():
        raise ValueError("Could not align neural component")
    return samples[aligned["_row"].astype(int).to_numpy()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--peak", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=500)
    args = parser.parse_args()

    base_frame, base = load_component(args.base)
    peak_frame, peak_raw = load_component(args.peak)
    peak = align(base_frame, peak_frame, peak_raw)
    hybrid = load_exact_horizon(args.hybrid, base_frame, 3, args.draws)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    policy = selection["policies"]["3"]["distribution"]
    if policy["kind"] != "soft":
        raise ValueError("This operational script expects the locked H3 soft policy")
    weights = [float(value) for value in policy["weights"]]
    arrays = [hybrid, base, peak]
    counts = [int(round(args.draws * weight)) for weight in weights]
    counts[0] += args.draws - sum(counts)
    samples = np.concatenate(
        [array[:, :count] for array, count in zip(arrays, counts) if count], axis=1
    )
    output = base_frame.copy()
    output["model"] = "DEEPST-INLA-H3-OPERATIONAL"
    output["predicted_mean"] = samples.mean(axis=1)
    output["predicted_median"] = np.median(samples, axis=1)
    for level, lower, upper in ((50, .25, .75), (80, .10, .90), (95, .025, .975)):
        output[f"lower_{level}"] = np.quantile(samples, lower, axis=1)
        output[f"upper_{level}"] = np.quantile(samples, upper, axis=1)
    args.output.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output / "predictions_h3.csv", index=False)
    np.savez_compressed(args.output / "predictive_samples_h3.npz", samples=samples)
    manifest = {
        "selection": portable_path(args.selection), "horizon": 3,
        "distribution_policy": policy, "draw_counts": counts,
        "rows": len(output), "draws": args.draws,
    }
    (args.output / "policy_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
