#!/usr/bin/env python3
"""Apply the original validation-locked final policy to one rolling vintage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_multiobjective_ensemble import run_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", type=Path, required=True)
    parser.add_argument("--peak", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=500)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    policies = {int(key): value for key, value in selection["policies"].items()}
    frame, samples, _, diagnostics = run_split(
        "test", args.deep, args.peak, args.hybrid, policies, args.draws, 100
    )
    frame["model"] = "DEEPST-INLA-ROLLING-REFIT"
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "predictions.csv", index=False)
    np.savez_compressed(args.output / "predictive_samples.npz", samples=samples)
    (args.output / "policy_manifest.json").write_text(
        json.dumps({
            "source_selection": str(args.selection.resolve()),
            "selection_split": selection["selection_split"],
            "policies": policies, "diagnostics": diagnostics,
        }, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
