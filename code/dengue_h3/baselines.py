from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import SnapshotData
from .evaluation import score_predictions


def run_baseline(
    name: str,
    data: SnapshotData,
    test_mask: np.ndarray | None,
    output_dir: str | Path,
    *,
    draws: int,
    seed: int,
    target_start: str | None = None,
    target_end: str | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    if (target_start is None) != (target_end is None):
        raise ValueError("target_start and target_end must be supplied together")
    if target_start is not None:
        selected_by_horizon = _exact_target_window(data, target_start, target_end)
        prediction_frames = []
        sample_blocks = []
        for selected in selected_by_horizon:
            historical = np.expm1(selected.seasonal[..., :5])
            samples = _sample_baseline(name, selected, historical, draws, rng)
            prediction_frames.append(_prediction_frame(name, selected, samples))
            sample_blocks.append(samples)
        origin_counts = {block.shape[0] for block in sample_blocks}
        if len(origin_counts) != 1:
            raise ValueError("Exact target window has unequal origin counts by horizon")
        predictions = pd.concat(prediction_frames, ignore_index=True)
        samples = np.concatenate(sample_blocks, axis=2)
    else:
        if test_mask is None:
            raise ValueError("test_mask is required without an exact target window")
        selected = data.subset(test_mask)
        historical = np.expm1(selected.seasonal[..., :5])
        samples = _sample_baseline(name, selected, historical, draws, rng)
        predictions = _prediction_frame(name, selected, samples)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    np.savez_compressed(output_dir / "predictive_samples.npz", samples=samples)
    score_predictions(predictions, samples).to_csv(
        output_dir / "metrics.csv", index=False
    )
    return output_dir


def _sample_baseline(
    name: str,
    data: SnapshotData,
    historical: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if name == "seasonal_naive":
        choices = rng.integers(
            0, historical.shape[-1], size=(*historical.shape[:-1], draws)
        )
        return np.take_along_axis(
            historical[..., None, :], choices[..., None], axis=-1
        )[..., 0].astype(np.float32)
    if name == "province_allocation":
        return _province_allocation(data, historical, draws, rng)
    raise ValueError("Baseline must be seasonal_naive or province_allocation")


def _exact_target_window(
    data: SnapshotData, target_start: str, target_end: str
) -> list[SnapshotData]:
    """Select each horizon by its own target date without changing row keys."""
    lower = pd.Timestamp(target_start)
    upper = pd.Timestamp(target_end)
    output = []
    for horizon_index, horizon in enumerate(data.horizons):
        dates = pd.to_datetime(data.target_dates[:, horizon_index])
        mask = np.asarray((dates >= lower) & (dates <= upper))
        output.append(
            SnapshotData(
                origins=data.origins[mask],
                units=data.units,
                provinces=data.provinces,
                horizons=(horizon,),
                local=data.local[mask],
                climate=data.climate[mask],
                seasonal=data.seasonal[mask, :, horizon_index : horizon_index + 1],
                targets=data.targets[mask, :, horizon_index : horizon_index + 1],
                population=data.population[mask, :, horizon_index : horizon_index + 1],
                target_dates=data.target_dates[mask, horizon_index : horizon_index + 1],
                outbreak_thresholds=data.outbreak_thresholds[
                    mask, :, horizon_index : horizon_index + 1
                ],
            )
        )
    return output


def _province_allocation(
    data: SnapshotData,
    historical: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    origins, nodes, horizons, _ = historical.shape
    samples = np.zeros((origins, nodes, horizons, draws), dtype=np.float32)
    provinces = np.asarray(data.provinces)
    for province in np.unique(provinces):
        node_indices = np.flatnonzero(provinces == province)
        province_history = historical[:, node_indices].sum(axis=1)
        choices = rng.integers(
            0, province_history.shape[-1], size=(origins, horizons, draws)
        )
        total_draws = np.take_along_axis(
            province_history[..., None, :], choices[..., None], axis=-1
        )[..., 0].astype(int)
        for origin in range(origins):
            for horizon in range(horizons):
                # Allocation weights must be available when the forecast is
                # issued. The third local channel is log1p population and the
                # final sequence position is the issue month. Target-month
                # population is retained only for incidence scoring.
                shares = np.expm1(
                    data.local[origin, node_indices, -1, 2]
                ).astype(float)
                shares /= shares.sum()
                allocated = rng.multinomial(
                    total_draws[origin, horizon], shares
                )
                samples[origin, node_indices, horizon] = allocated.T
    return samples


def _prediction_frame(
    name: str,
    data: SnapshotData,
    samples: np.ndarray,
) -> pd.DataFrame:
    origins, nodes, horizons, _ = samples.shape
    thresholds = data.outbreak_thresholds
    quantiles = np.quantile(
        samples, [0.25, 0.75, 0.10, 0.90, 0.025, 0.975], axis=-1
    )
    target_dates = np.repeat(data.target_dates[:, None, :], nodes, axis=1)
    return pd.DataFrame(
        {
            "model": name,
            "fcode": np.tile(np.repeat(np.asarray(data.units), horizons), origins),
            "issue_date": np.repeat(data.origins.to_numpy(), nodes * horizons),
            "target_date": target_dates.reshape(-1),
            "horizon": np.tile(np.asarray(data.horizons), origins * nodes),
            "target_cases": data.targets.reshape(-1),
            "population": data.population.reshape(-1),
            "predicted_mean": samples.mean(axis=-1).reshape(-1),
            "predicted_median": np.median(samples, axis=-1).reshape(-1),
            "lower_50": quantiles[0].reshape(-1),
            "upper_50": quantiles[1].reshape(-1),
            "lower_80": quantiles[2].reshape(-1),
            "upper_80": quantiles[3].reshape(-1),
            "lower_95": quantiles[4].reshape(-1),
            "upper_95": quantiles[5].reshape(-1),
            "outbreak_threshold": thresholds.reshape(-1),
            "outbreak_probability": (
                samples > thresholds[..., None]
            ).mean(axis=-1).reshape(-1),
        }
    )
