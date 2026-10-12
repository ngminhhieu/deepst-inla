from pathlib import Path
import sys

import pandas as pd


PIPELINE = Path(__file__).resolve().parents[1] / "paper_pipeline"
sys.path.insert(0, str(PIPELINE))

from analyze_spatial_support import allocate, nested_maps  # noqa: E402


def test_allocate_preserves_requested_total_and_provinces():
    result = allocate(8, {"a": 10, "b": 4, "c": 2})
    assert sum(result.values()) == 8
    assert all(value >= 1 for value in result.values())
    assert all(value <= {"a": 10, "b": 4, "c": 2}[key] for key, value in result.items())


def test_nested_maps_never_split_previous_group():
    units = ["1", "2", "3", "4", "5", "6"]
    monthly = pd.DataFrame({
        "fcode": units, "l1_code": ["a", "a", "a", "b", "b", "b"]
    })
    centroids = pd.DataFrame({
        "fcode": units, "longitude": [0, 1, 2, 10, 11, 12], "latitude": [0] * 6
    })
    adjacency = pd.DataFrame({
        "from": ["1", "2", "4", "5"], "to": ["2", "3", "5", "6"]
    })
    maps = nested_maps(units, monthly, centroids, adjacency, (4,))
    assert maps["commune_554"].aggregate_id.nunique() == 6
    assert maps["support_4"].aggregate_id.nunique() == 4
    assert maps["province_2"].aggregate_id.nunique() == 2
