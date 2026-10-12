import pandas as pd

from scripts.prepare_qc_data import audit_communes


def _panel() -> pd.DataFrame:
    rows = []
    for fcode, population, area, cases in (
        ("valid", 20_000, 20.0, 2),
        ("invalid", 50, 100.0, 10),
    ):
        for month in (1, 2, 3):
            rows.append(
                {
                    "fcode": fcode,
                    "year": 2020,
                    "month": month,
                    "obs_dengue_cases": cases,
                    "population": population,
                    "area": area,
                    "t2m_avg": 28.0,
                    "rh_avg": 80.0,
                    "tp_accum": 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_audit_excludes_implausible_population_but_retains_valid_panel():
    names = pd.DataFrame(
        {
            "fcode": ["valid", "invalid"],
            "region_name": ["Valid", "Invalid"],
            "province_name": ["P", "P"],
        }
    )
    report = audit_communes(
        _panel(),
        names,
        minimum_population=1_000,
        minimum_density=5,
        maximum_monthly_incidence=5_000,
    ).set_index("fcode")

    assert not bool(report.loc["valid", "qc_exclude"])
    assert bool(report.loc["invalid", "qc_exclude"])
    assert "population_below_minimum" in report.loc["invalid", "qc_reasons"]
    assert "population_density_below_minimum" in report.loc["invalid", "qc_reasons"]
    assert "monthly_incidence_above_maximum" in report.loc["invalid", "qc_reasons"]
