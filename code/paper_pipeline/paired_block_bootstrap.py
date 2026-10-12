#!/usr/bin/env python3
"""Paired target-month block bootstrap for two probabilistic forecast systems.

All communes and horizons belonging to a target calendar month remain in the
same resampling block. CRPS and Brier are aggregated from row-level losses;
F1 is correctly recomputed from resampled TP/FP/FN counts rather than averaged
as if it were a decomposable row-level metric.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["fcode", "target_date", "horizon"]
HORIZONS = (1, 2, 3)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crps(observed: np.ndarray, samples: np.ndarray) -> np.ndarray:
    ordered = np.sort(samples, axis=1)
    count = ordered.shape[1]
    weights = 2 * np.arange(1, count + 1) - count - 1
    pairwise = (ordered * weights[None]).sum(axis=1) / (count * count)
    return np.abs(samples - observed[:, None]).mean(axis=1) - pairwise


def load_samples(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        if "samples" not in archive:
            raise ValueError(f"{path} does not contain a 'samples' array")
        return archive["samples"]


def add_row_statistics(
    frame: pd.DataFrame,
    sample_path: Path,
    *,
    label_thresholds: pd.Series | None = None,
    probability_from_samples: bool = False,
    decision_threshold: float | None = None,
) -> pd.DataFrame:
    samples = load_samples(sample_path)
    horizons = sorted(int(value) for value in frame["horizon"].unique())
    if tuple(horizons) != HORIZONS or samples.shape[2] != len(HORIZONS):
        raise ValueError(f"Expected H1-H3 sample cube, found {samples.shape}")
    output = frame.copy()
    output["crps_loss"] = np.nan
    output["brier_loss"] = np.nan
    if label_thresholds is not None:
        output["label_threshold"] = label_thresholds.to_numpy(float)
    else:
        output["label_threshold"] = output["outbreak_threshold"].to_numpy(float)

    for horizon_index, horizon in enumerate(horizons):
        indices = np.flatnonzero(output["horizon"].to_numpy(int) == horizon)
        part = output.iloc[indices]
        values = samples[:, :, horizon_index, :].reshape(-1, samples.shape[-1])
        if len(part) != len(values):
            raise ValueError(
                f"H{horizon}: {len(part)} prediction rows do not match "
                f"{len(values)} sample rows"
            )
        population = part["population"].to_numpy(float)
        observed_incidence = part["target_cases"].to_numpy(float) / population * 100_000.0
        sample_incidence = values / population[:, None] * 100_000.0
        output.iloc[indices, output.columns.get_loc("crps_loss")] = crps(
            observed_incidence, sample_incidence
        )
        thresholds = part["label_threshold"].to_numpy(float)
        if probability_from_samples:
            probabilities = (values > thresholds[:, None]).mean(axis=1)
            output.iloc[indices, output.columns.get_loc("outbreak_probability")] = probabilities

    observed = (
        output["target_cases"].to_numpy(float)
        > output["label_threshold"].to_numpy(float)
    )
    probability = output["outbreak_probability"].to_numpy(float)
    output["brier_loss"] = np.square(probability - observed)
    if decision_threshold is None:
        if "outbreak_decision_threshold" in output:
            cutoff = output["outbreak_decision_threshold"].to_numpy(float)
        else:
            cutoff = np.full(len(output), 0.5)
    else:
        cutoff = np.full(len(output), float(decision_threshold))
    alert = probability >= cutoff
    output["tp"] = observed & alert
    output["fp"] = ~observed & alert
    output["fn"] = observed & ~alert
    return output


def monthly_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    frame = frame.copy()
    frame["month"] = pd.to_datetime(frame["target_date"]).dt.to_period("M").astype(str)
    months = sorted(frame["month"].unique())
    if len(months) < 2:
        raise ValueError("At least two target-month blocks are required")
    output: dict[str, np.ndarray] = {"months": np.asarray(months)}
    for column in ("crps_loss", "brier_loss", "tp", "fp", "fn"):
        table = (
            frame.groupby(["month", "horizon"])[column]
            .sum()
            .unstack("horizon")
            .reindex(index=months, columns=HORIZONS)
        )
        if table.isna().any().any():
            raise ValueError(f"Incomplete month/horizon blocks for {column}")
        output[column] = table.to_numpy(float)
    counts = (
        frame.groupby(["month", "horizon"])
        .size()
        .unstack("horizon")
        .reindex(index=months, columns=HORIZONS)
    )
    output["n"] = counts.to_numpy(float)
    return output


def metric(blocks: dict[str, np.ndarray], selected: np.ndarray, name: str) -> np.ndarray:
    if name in ("crps", "brier"):
        values = blocks[f"{name}_loss"][selected].sum(axis=0) / blocks["n"][selected].sum(axis=0)
        return values
    tp = blocks["tp"][selected].sum(axis=0)
    fp = blocks["fp"][selected].sum(axis=0)
    fn = blocks["fn"][selected].sum(axis=0)
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a", type=Path, help="Prediction CSV for model A")
    parser.add_argument("b", type=Path, help="Prediction CSV for model B")
    parser.add_argument("--samples-a", type=Path, required=True)
    parser.add_argument("--samples-b", type=Path, required=True)
    parser.add_argument("--name-a", default="model_a")
    parser.add_argument("--name-b", default="model_b")
    parser.add_argument(
        "--recompute-b-probability",
        action="store_true",
        help="Recompute model B exceedance probability using model A thresholds.",
    )
    parser.add_argument(
        "--decision-threshold-b",
        type=float,
        default=None,
        help=(
            "Optional fixed cutoff for model B. By default, use the validation-"
            "selected outbreak_decision_threshold stored in its prediction file."
        ),
    )
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--out", type=Path, default=Path("results/pending/bootstrap_pairwise.csv"))
    args = parser.parse_args()

    a = pd.read_csv(args.a, dtype={"fcode": str})
    b = pd.read_csv(args.b, dtype={"fcode": str})
    for frame, path in ((a, args.a), (b, args.b)):
        missing = set(KEYS + ["target_cases", "population", "outbreak_threshold", "outbreak_probability"]) - set(frame)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{path} contains duplicate forecast keys")
    a = a.sort_values(KEYS).reset_index(drop=True)
    b = b.sort_values(KEYS).reset_index(drop=True)
    if not a[KEYS].equals(b[KEYS]):
        raise ValueError("Prediction files do not contain identical locked keys")
    for column in ("target_cases", "population"):
        if not np.allclose(a[column], b[column], rtol=1e-7, atol=1e-7):
            raise ValueError(f"Prediction files disagree on {column}")

    # Restore each file's native generation order before aligning row statistics.
    a_native = pd.read_csv(args.a, dtype={"fcode": str})
    b_native = pd.read_csv(args.b, dtype={"fcode": str})
    a_stats = add_row_statistics(a_native, args.samples_a)
    local_thresholds = a_stats.set_index(KEYS)["label_threshold"].reindex(
        pd.MultiIndex.from_frame(b_native[KEYS])
    )
    if local_thresholds.isna().any():
        raise ValueError("Could not align model A outbreak thresholds to model B")
    b_stats = add_row_statistics(
        b_native,
        args.samples_b,
        label_thresholds=local_thresholds.reset_index(drop=True),
        probability_from_samples=args.recompute_b_probability,
        decision_threshold=args.decision_threshold_b,
    )
    a_stats = a_stats.sort_values(KEYS).reset_index(drop=True)
    b_stats = b_stats.sort_values(KEYS).reset_index(drop=True)
    blocks_a = monthly_arrays(a_stats)
    blocks_b = monthly_arrays(b_stats)
    if not np.array_equal(blocks_a["months"], blocks_b["months"]):
        raise ValueError("Models have different target-month blocks")

    month_count = len(blocks_a["months"])
    full = np.arange(month_count)
    rng = np.random.default_rng(args.seed)
    rows = []
    for metric_name in ("crps", "brier", "f1"):
        point_a = metric(blocks_a, full, metric_name)
        point_b = metric(blocks_b, full, metric_name)
        boot = np.empty((args.reps, len(HORIZONS)), dtype=float)
        for rep in range(args.reps):
            selected = rng.integers(0, month_count, size=month_count)
            boot[rep] = metric(blocks_a, selected, metric_name) - metric(
                blocks_b, selected, metric_name
            )
        for horizon_index, horizon in enumerate(HORIZONS):
            delta = boot[:, horizon_index]
            rows.append(
                {
                    "model_a": args.name_a,
                    "model_b": args.name_b,
                    "metric": metric_name,
                    "scope": f"H{horizon}",
                    "estimate_a": point_a[horizon_index],
                    "estimate_b": point_b[horizon_index],
                    "difference_a_minus_b": point_a[horizon_index] - point_b[horizon_index],
                    "ci025": np.quantile(delta, 0.025),
                    "ci975": np.quantile(delta, 0.975),
                    "probability_difference_below_zero": np.mean(delta < 0),
                    "reps": args.reps,
                    "blocks": month_count,
                }
            )
        macro = boot.mean(axis=1)
        rows.append(
            {
                "model_a": args.name_a,
                "model_b": args.name_b,
                "metric": metric_name,
                "scope": "mean_H1_H3",
                "estimate_a": point_a.mean(),
                "estimate_b": point_b.mean(),
                "difference_a_minus_b": point_a.mean() - point_b.mean(),
                "ci025": np.quantile(macro, 0.025),
                "ci975": np.quantile(macro, 0.975),
                "probability_difference_below_zero": np.mean(macro < 0),
                "reps": args.reps,
                "blocks": month_count,
            }
        )
    output = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    provenance = args.out.with_suffix(".provenance.txt")
    provenance.write_text(
        "\n".join(
            [
                "Paired target-calendar-month block bootstrap.",
                f"model_a_predictions={args.name_a}/{args.a.name}",
                f"model_a_predictions_sha256={sha256(args.a)}",
                f"model_a_samples={args.name_a}/{args.samples_a.name}",
                f"model_a_samples_sha256={sha256(args.samples_a)}",
                f"model_b_predictions={args.name_b}/{args.b.name}",
                f"model_b_predictions_sha256={sha256(args.b)}",
                f"model_b_samples={args.name_b}/{args.samples_b.name}",
                f"model_b_samples_sha256={sha256(args.samples_b)}",
                f"reps={args.reps}",
                f"seed={args.seed}",
                "difference_direction=model_a_minus_model_b",
                "lower_is_better=crps,brier",
                "higher_is_better=f1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
