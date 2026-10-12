import numpy as np
import pandas as pd

from dengue_h3.baselines import _exact_target_window, _province_allocation
from dengue_h3.data import SnapshotData


def test_parent_allocation_uses_issue_month_population() -> None:
    local = np.zeros((1, 2, 1, 5), dtype=np.float32)
    local[0, :, -1, 2] = np.log1p([900.0, 100.0])
    snapshot = SnapshotData(
        origins=pd.DatetimeIndex(["2020-12-01"]),
        units=("a", "b"),
        provinces=("p", "p"),
        horizons=(1,),
        local=local,
        climate=np.zeros((1, 2, 1, 1), dtype=np.float32),
        seasonal=np.zeros((1, 2, 1, 5), dtype=np.float32),
        targets=np.zeros((1, 2, 1), dtype=np.float32),
        # Reverse target-month population so the test detects future leakage.
        population=np.asarray([[[100.0], [900.0]]], dtype=np.float32),
        target_dates=np.asarray([[np.datetime64("2021-01-01")]]),
        outbreak_thresholds=np.zeros((1, 2, 1), dtype=np.float32),
    )
    history = np.full((1, 2, 1, 5), 50.0, dtype=np.float32)
    draws = _province_allocation(
        snapshot, history, 2_000, np.random.default_rng(1234)
    )
    first_share = draws[0, 0, 0].sum() / draws[0, :, 0].sum()
    assert first_share > 0.85


def test_exact_target_window_uses_horizon_specific_origins() -> None:
    origins = pd.date_range("2020-10-01", periods=5, freq="MS")
    target_dates = np.asarray(
        [
            [origin + pd.DateOffset(months=horizon) for horizon in (1, 2, 3)]
            for origin in origins
        ],
        dtype="datetime64[ns]",
    )
    snapshot = SnapshotData(
        origins=origins,
        units=("a",),
        provinces=("p",),
        horizons=(1, 2, 3),
        local=np.zeros((5, 1, 2, 3), dtype=np.float32),
        climate=np.zeros((5, 1, 2, 1), dtype=np.float32),
        seasonal=np.zeros((5, 1, 3, 5), dtype=np.float32),
        targets=np.zeros((5, 1, 3), dtype=np.float32),
        population=np.ones((5, 1, 3), dtype=np.float32),
        target_dates=target_dates,
        outbreak_thresholds=np.zeros((5, 1, 3), dtype=np.float32),
    )

    selected = _exact_target_window(snapshot, "2021-01-01", "2021-02-01")

    assert [list(item.origins.strftime("%Y-%m")) for item in selected] == [
        ["2020-12", "2021-01"],
        ["2020-11", "2020-12"],
        ["2020-10", "2020-11"],
    ]
    assert all(item.targets.shape == (2, 1, 1) for item in selected)
