#!/usr/bin/env python3
"""Score validation predictions already exported by a deep experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dengue_h3.evaluation import score_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    predictions = pd.read_csv(args.experiment_dir / "validation_predictions.csv")
    samples = np.load(args.experiment_dir / "validation_predictive_samples.npz")["samples"]
    score_predictions(predictions, samples, split="validation").to_csv(
        args.experiment_dir / "validation_metrics.csv", index=False
    )


if __name__ == "__main__":
    main()
