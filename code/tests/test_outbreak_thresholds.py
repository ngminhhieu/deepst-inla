import numpy as np
import pandas as pd

from dengue_h3.data import (
    _moh_outbreak_threshold_lookup,
    _outbreak_thresholds_for_targets,
    _outbreak_threshold_lookup,
    load_outbreak_thresholds,
)


def test_external_thresholds_are_mapped_by_commune_and_target_month(tmp_path):
    path = tmp_path / "thresholds.csv"
    pd.DataFrame(
        {
            "fcode": ["001", "002"],
            "year": [2021, 2021],
            "month": [2, 2],
            "epi_threshold": [7, 11],
        }
    ).to_csv(path, index=False)
    frame = load_outbreak_thresholds(path)
    lookup = _outbreak_threshold_lookup(frame, ("001", "002"))
    values = _outbreak_thresholds_for_targets(
        pd.date_range("2021-01-01", periods=3, freq="MS"),
        0,
        (1, 2),
        ("001", "002"),
        lookup,
    )
    np.testing.assert_array_equal(values[:, 0], [7, 11])
    assert np.isinf(values[:, 1]).all()


def test_missing_early_moh_thresholds_are_treated_as_unavailable(tmp_path):
    path = tmp_path / "thresholds.csv"
    pd.DataFrame(
        {
            "fcode": ["001", "001"],
            "year": [2015, 2016],
            "month": [1, 1],
            "epi_threshold": [np.nan, 7.0],
        }
    ).to_csv(path, index=False)

    frame = load_outbreak_thresholds(path)

    assert len(frame) == 1
    assert frame.iloc[0]["outbreak_threshold"] == 7.0


def test_internal_moh_fallback_excludes_and_replaces_outbreak_year():
    dates = pd.date_range("2010-01-01", "2016-01-01", freq="12MS")
    frame = pd.DataFrame(
        {
            "fcode": "001",
            "date": dates,
            "cases": [1.0, 2.0, 1.0, 2.0, 1.0, 100.0, 3.0],
        }
    )

    lookup = _moh_outbreak_threshold_lookup(frame, dates, ("001",))

    # For 2016, 2015 is detected as an outbreak and replaced by 2010.
    expected = np.mean([1, 2, 1, 2, 1]) + 2 * np.std([1, 2, 1, 2, 1], ddof=1)
    assert np.isclose(lookup[("001", pd.Timestamp("2016-01-01"))], expected)
    # For 2012, only two prior years exist, so the early-history fallback is used.
    early = np.mean([2, 1]) + 2 * np.std([2, 1], ddof=1)
    assert np.isclose(lookup[("001", pd.Timestamp("2012-01-01"))], early)
