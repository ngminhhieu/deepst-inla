import numpy as np
import pandas as pd
import pytest

from paper_pipeline.prepare_province_scale import aggregate_panel
from paper_pipeline.analyze_province_scale import apply_platt, fit_platt, select_f1_cutoff


def _source():
    rows = []
    for unit, province, population, cases in (("a", "1", 100, 2), ("b", "1", 300, 5)):
        for month in (1, 2):
            row = {
                "fcode": unit, "l1_code": province, "year": 2020, "month": month,
                "obs_dengue_cases": cases, "population": population, "area": 1,
            }
            row.update({column: float(cases) for column in (
                "ws_avg", "ws_min", "ws_max", "t2m_avg", "t2m_min", "t2m_max",
                "rh_avg", "rh_min", "rh_max", "tp_accum")})
            rows.append(row)
    return pd.DataFrame(rows)


def test_aggregate_panel_sums_counts_and_population_weights_covariates():
    output = aggregate_panel(_source())
    assert len(output) == 2
    assert output.obs_dengue_cases.tolist() == [7.0, 7.0]
    assert output.population.tolist() == [400.0, 400.0]
    assert np.allclose(output.t2m_avg, 4.25)


def test_aggregate_panel_rejects_duplicate_unit_months():
    source = pd.concat([_source(), _source().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_panel(source)


def test_platt_and_cutoff_use_only_supplied_validation_arrays():
    probability = np.array([0.05, 0.10, 0.20, 0.70, 0.80, 0.95])
    labels = np.array([0, 0, 0, 1, 1, 1])
    intercept, slope = fit_platt(probability, labels)
    calibrated = apply_platt(probability, intercept, slope)
    assert slope > 0
    assert calibrated[-1] > calibrated[0]
    cutoff = select_f1_cutoff(labels, calibrated)
    assert 0 < cutoff < 1
