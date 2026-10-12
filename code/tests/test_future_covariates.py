from pathlib import Path

import pandas as pd
import pytest

from dengue_h3.data import load_future_covariates


def test_future_covariates_require_target_after_issue(tmp_path: Path):
    path = tmp_path / "future.csv"
    pd.DataFrame(
        {
            "fcode": ["1"],
            "issue_date": ["2024-01-01"],
            "target_date": ["2024-01-01"],
            "rain_forecast": [12.0],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="target_date after issue_date"):
        load_future_covariates(path, ["rain_forecast"])


def test_future_covariates_normalise_month_keys(tmp_path: Path):
    path = tmp_path / "future.csv"
    pd.DataFrame(
        {
            "fcode": ["001"],
            "issue_date": ["2024-01-15"],
            "target_date": ["2024-03-28"],
            "rain_forecast": [12.0],
        }
    ).to_csv(path, index=False)
    result = load_future_covariates(path, ["rain_forecast"])
    assert result.loc[0, "fcode"] == "001"
    assert result.loc[0, "issue_date"] == pd.Timestamp("2024-01-01")
    assert result.loc[0, "target_date"] == pd.Timestamp("2024-03-01")
