import numpy as np
import pandas as pd

from dengue_h3.evaluation import outbreak_classification_metrics, score_predictions


def test_outbreak_classification_reports_precision_specificity_recall_and_f1():
    result = outbreak_classification_metrics(
        observed_cases=np.array([12, 8, 15, 5]),
        outbreak_threshold=np.array([10, 10, 10, 10]),
        outbreak_probability=np.array([0.8, 0.7, 0.4, 0.1]),
        decision_threshold=0.5,
    )
    assert result["outbreak_tp"] == 1
    assert result["outbreak_fp"] == 1
    assert result["outbreak_fn"] == 1
    assert result["outbreak_tn"] == 1
    assert result["outbreak_precision"] == 0.5
    assert result["outbreak_specificity"] == 0.5
    assert result["outbreak_recall"] == 0.5
    assert result["outbreak_sensitivity"] == 0.5
    assert result["outbreak_true_negative_rate"] == 0.5
    assert result["outbreak_f1"] == 0.5


def test_no_predicted_outbreaks_have_zero_precision_recall_and_perfect_specificity():
    result = outbreak_classification_metrics(
        observed_cases=np.array([12, 8]),
        outbreak_threshold=np.array([10, 10]),
        outbreak_probability=np.array([0.2, 0.1]),
    )
    assert result["outbreak_precision"] == 0.0
    assert result["outbreak_specificity"] == 1.0
    assert result["outbreak_recall"] == 0.0
    assert result["outbreak_sensitivity"] == 0.0
    assert result["outbreak_true_negative_rate"] == 1.0
    assert result["outbreak_f1"] == 0.0


def test_score_predictions_accepts_horizon_specific_alert_thresholds_and_split():
    predictions = pd.DataFrame(
        {
            "model": ["test-model", "test-model"],
            "horizon": [1, 2],
            "target_cases": [2, 2],
            "population": [100_000, 100_000],
            "outbreak_threshold": [1, 1],
            "outbreak_probability": [0.6, 0.6],
        }
    )
    samples = np.full((1, 1, 2, 4), 2.0)
    metrics = score_predictions(
        predictions,
        samples,
        decision_threshold={1: 0.5, 2: 0.7},
        split="validation",
    ).set_index("horizon")

    assert set(metrics["split"]) == {"validation"}
    assert metrics.loc[1, "outbreak_tp"] == 1
    assert metrics.loc[2, "outbreak_fn"] == 1
    assert metrics.loc[1, "outbreak_decision_threshold"] == 0.5
    assert metrics.loc[2, "outbreak_decision_threshold"] == 0.7
    assert "outbreak_precision" in metrics.columns
    assert "outbreak_specificity" in metrics.columns
    assert "outbreak_recall" in metrics.columns
    assert "outbreak_sensitivity" in metrics.columns
