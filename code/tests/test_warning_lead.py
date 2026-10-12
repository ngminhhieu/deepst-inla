import pandas as pd

from scripts.evaluate_warning_lead_time import warning_lead_metrics


def test_warning_lead_uses_earliest_alert_across_horizons():
    rows = []
    for horizon, issue, probability in (
        (3, "2024-01-01", 0.6),
        (2, "2024-02-01", 0.8),
        (1, "2024-03-01", 0.9),
    ):
        rows.append(
            {
                "model": "M",
                "fcode": "1",
                "issue_date": issue,
                "target_date": "2024-04-01",
                "horizon": horizon,
                "target_cases": 20,
                "outbreak_threshold": 10,
                "outbreak_probability": probability,
            }
        )
    result = warning_lead_metrics(pd.DataFrame(rows))
    assert result["observed_outbreak_target_months"] == 1
    assert result["detected_outbreak_target_months"] == 1
    assert result["median_warning_lead_months_detected"] == 3
    assert result["share_detected_with_lead_ge_3m"] == 1.0
