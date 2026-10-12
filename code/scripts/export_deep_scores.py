#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("representations", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--features",
        choices=("location", "location_scale"),
        default="location",
    )
    parser.add_argument("--minimum-scale", type=float, default=0.05)
    parser.add_argument("--maximum-scale", type=float, default=3.0)
    args = parser.parse_args()

    frame = pd.read_csv(args.representations, dtype={"fcode": str})
    z_columns = [column for column in frame if column.startswith("z_")]
    if not z_columns:
        raise ValueError("No representation columns were found")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    latent = frame[z_columns].to_numpy(np.float32)
    location = np.empty(len(frame), dtype=np.float32)
    scale = np.empty(len(frame), dtype=np.float32)

    for horizon_index, horizon in enumerate(sorted(frame["horizon"].unique())):
        selected = frame["horizon"].eq(horizon).to_numpy()
        mean_weight = state[f"mean_heads.{horizon_index}.weight"].numpy().reshape(-1)
        mean_bias = float(state[f"mean_heads.{horizon_index}.bias"].numpy()[0])
        scale_weight = state[
            f"dispersion_heads.{horizon_index}.weight"
        ].numpy().reshape(-1)
        scale_bias = float(
            state[f"dispersion_heads.{horizon_index}.bias"].numpy()[0]
        )
        location[selected] = latent[selected] @ mean_weight + mean_bias
        raw_scale = latent[selected] @ scale_weight + scale_bias
        scale[selected] = np.minimum(
            np.logaddexp(0.0, raw_scale) + args.minimum_scale,
            args.maximum_scale,
        )

    required = [
        "split", "fcode", "issue_date", "target_date", "horizon",
        "target_cases", "population", "outbreak_threshold",
    ]
    output = frame[required].copy()
    output["z_01"] = location
    if args.features == "location_scale":
        output["z_02"] = scale
    output["deep_expected_cases"] = np.maximum(
        np.exp(location + 0.5 * np.square(scale)) - 1.0,
        0.0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
