import numpy as np
import pandas as pd

from scripts.build_open_covariates import causal_fill


def test_causal_fill_never_backfills_from_the_future():
    frame = pd.DataFrame(
        {
            "fcode": ["a", "a", "a", "b", "b"],
            "date": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-03-01", "2020-01-01", "2020-02-01"]
            ),
            "value": [np.nan, 2.0, np.nan, 4.0, np.nan],
        }
    )
    output = causal_fill(frame, ["value"])
    assert output.loc[output.fcode.eq("a"), "value"].tolist() == [0.0, 2.0, 2.0]
    assert output.loc[output.fcode.eq("b"), "value"].tolist() == [4.0, 4.0]
