#!/usr/bin/env python3
"""Target-month block-bootstrap intervals for every fixed-split report model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS = (1, 2, 3)
METRICS = ("mae", "rmse", "crps", "outbreak_brier", "outbreak_precision",
           "outbreak_specificity", "outbreak_sensitivity", "outbreak_f1")


def crps(observed: np.ndarray, samples: np.ndarray) -> np.ndarray:
    ordered = np.sort(samples, axis=1)
    count = samples.shape[1]
    weights = 2 * np.arange(1, count + 1) - count - 1
    return (np.abs(samples - observed[:, None]).mean(axis=1)
            - (ordered * weights).sum(axis=1) / (count * count))


def sample_matrices(directory: Path, frame: pd.DataFrame) -> dict[int, np.ndarray]:
    archive = directory / "predictive_samples.npz"
    if archive.exists():
        with np.load(archive) as data:
            cube = data["samples"]
        return {
            horizon: cube[:, :, index, :].reshape(-1, cube.shape[-1])
            for index, horizon in enumerate(HORIZONS)
        }
    output = {}
    for horizon in HORIZONS:
        shape = pd.read_csv(directory / f"samples_H{horizon}_shape.csv").iloc[0]
        rows, draws = int(shape["rows"]), int(shape["draws"])
        raw = np.fromfile(directory / f"samples_H{horizon}.bin", dtype=np.float64)
        output[horizon] = raw.reshape((rows, draws), order="F")
    return output


def row_statistics(directory: Path) -> pd.DataFrame:
    frame = pd.read_csv(directory / "predictions.csv", dtype={"fcode": str})
    samples = sample_matrices(directory, frame)
    pieces = []
    for horizon in HORIZONS:
        part = frame.loc[frame["horizon"].astype(int).eq(horizon)].copy()
        values = samples[horizon]
        if len(part) != len(values):
            raise ValueError(f"{directory} H{horizon}: predictions/samples row mismatch")
        population = part["population"].to_numpy(float)
        observed = part["target_cases"].to_numpy(float)
        observed_incidence = observed / population * 100_000
        sampled_incidence = values / population[:, None] * 100_000
        error = sampled_incidence.mean(axis=1) - observed_incidence
        outbreak = observed > part["outbreak_threshold"].to_numpy(float)
        probability = part["outbreak_probability"].to_numpy(float)
        cutoff = (part["outbreak_decision_threshold"].to_numpy(float)
                  if "outbreak_decision_threshold" in part else np.full(len(part), .5))
        alert = probability >= cutoff
        part["abs_error"] = np.abs(error)
        part["squared_error"] = np.square(error)
        part["crps_loss"] = crps(observed_incidence, sampled_incidence)
        part["brier_loss"] = np.square(probability - outbreak)
        part["tp"] = outbreak & alert
        part["tn"] = ~outbreak & ~alert
        part["fp"] = ~outbreak & alert
        part["fn"] = outbreak & ~alert
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def month_blocks(frame: pd.DataFrame) -> tuple[list[str], dict[str, np.ndarray]]:
    frame = frame.copy()
    frame["month"] = pd.to_datetime(frame["target_date"]).dt.to_period("M").astype(str)
    months = sorted(frame["month"].unique())
    columns = ("abs_error", "squared_error", "crps_loss", "brier_loss",
               "tp", "tn", "fp", "fn")
    arrays = {}
    for column in columns:
        table = (frame.groupby(["month", "horizon"])[column].sum().unstack()
                 .reindex(index=months, columns=HORIZONS))
        if table.isna().any().any():
            raise ValueError(f"Incomplete month/horizon blocks for {column}")
        arrays[column] = table.to_numpy(float)
    arrays["n"] = (frame.groupby(["month", "horizon"]).size().unstack()
                   .reindex(index=months, columns=HORIZONS).to_numpy(float))
    return months, arrays


def ratios(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def evaluate(blocks: dict[str, np.ndarray], selected: np.ndarray) -> dict[str, np.ndarray]:
    total = {key: value[selected].sum(axis=0) for key, value in blocks.items()}
    return {
        "mae": total["abs_error"] / total["n"],
        "rmse": np.sqrt(total["squared_error"] / total["n"]),
        "crps": total["crps_loss"] / total["n"],
        "outbreak_brier": total["brier_loss"] / total["n"],
        "outbreak_precision": ratios(total["tp"], total["tp"] + total["fp"]),
        "outbreak_specificity": ratios(total["tn"], total["tn"] + total["fp"]),
        "outbreak_sensitivity": ratios(total["tp"], total["tp"] + total["fn"]),
        "outbreak_f1": ratios(2 * total["tp"], 2 * total["tp"] + total["fp"] + total["fn"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="LABEL=directory")
    parser.add_argument("--reference", default="DeepST-INLA")
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = dict(item.split("=", 1) for item in args.model)
    blocks_by_model = {}
    common_months = None
    for label, path in models.items():
        months, blocks = month_blocks(row_statistics(Path(path)))
        if common_months is not None and months != common_months:
            raise ValueError(f"{label}: target months differ from other models")
        common_months = months
        blocks_by_model[label] = blocks
    rng = np.random.default_rng(args.seed)
    selections = rng.integers(0, len(common_months), size=(args.reps, len(common_months)))
    interval_rows, replicate_rows = [], []
    for label, blocks in blocks_by_model.items():
        point = evaluate(blocks, np.arange(len(common_months)))
        boot = {metric: np.empty((args.reps, 3)) for metric in METRICS}
        for index, selected in enumerate(selections):
            result = evaluate(blocks, selected)
            for metric in METRICS:
                boot[metric][index] = result[metric]
        for metric in METRICS:
            for hindex, scope in enumerate(("H1", "H2", "H3", "mean_H1_H3")):
                values = boot[metric][:, hindex] if hindex < 3 else boot[metric].mean(axis=1)
                estimate = point[metric][hindex] if hindex < 3 else point[metric].mean()
                interval_rows.append({"model": label, "metric": metric, "scope": scope,
                                      "estimate": estimate, "bootstrap_sd": values.std(ddof=1),
                                      "ci025": np.quantile(values, .025),
                                      "ci975": np.quantile(values, .975),
                                      "reps": args.reps, "blocks": len(common_months)})
                if scope == "mean_H1_H3":
                    replicate_rows.extend({"replicate": rep + 1, "model": label,
                                           "metric": metric, "value": value}
                                          for rep, value in enumerate(values))
    intervals = pd.DataFrame(interval_rows)
    replicates = pd.DataFrame(replicate_rows)
    reference = replicates.loc[replicates.model.eq(args.reference), ["replicate", "metric", "value"]]
    reference = reference.rename(columns={"value": "reference_value"})
    contrasts = replicates.merge(reference, on=["replicate", "metric"])
    contrasts["difference_vs_reference"] = contrasts["value"] - contrasts["reference_value"]
    contrasts = contrasts.loc[~contrasts.model.eq(args.reference)]
    contrast_summary = (contrasts.groupby(["model", "metric"])["difference_vs_reference"]
                        .agg(estimate="mean", ci025=lambda x: x.quantile(.025),
                             ci975=lambda x: x.quantile(.975),
                             probability_below_zero=lambda x: (x < 0).mean()).reset_index())
    args.output.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(args.output / "model_metric_intervals.csv", index=False)
    contrast_summary.to_csv(args.output / "paired_contrasts_vs_deepst_inla.csv", index=False)
    replicates.to_csv(args.output / "bootstrap_replicates_mean_H1_H3.csv", index=False)


if __name__ == "__main__":
    main()
