#!/usr/bin/env python3
"""Create a compact audit report for the fixed-model retrain and rolling CV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


LABELS = {
    "current_probability": "Current calibrated probability",
    "risk_abs": "v1 absolute-risk threshold",
    "risk_z": "v1 z-score threshold",
    "pred_mean": "Forecast mean > outbreak threshold",
    "two_of_three": "At least 2 of 3",
    "three_of_three": "All 3 methods",
}


def add_table(document: Document, frame: pd.DataFrame) -> None:
    table = document.add_table(rows=1, cols=len(frame.columns)); table.style = "Table Grid"
    for cell, column in zip(table.rows[0].cells, frame.columns): cell.text = str(column)
    for values in frame.itertuples(index=False, name=None):
        for cell, value in zip(table.add_row().cells, values):
            cell.text = f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs: run.font.size = Pt(8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--previous-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    final = args.run_root / "DEEPST-INLA"
    reports = args.run_root / "outbreak_reports"
    new = pd.read_csv(final / "metrics.csv")
    old = pd.read_csv(args.previous_model / "metrics.csv")
    comparison = []
    for label, data in (("Previous saved run", old), ("Fresh retrain", new)):
        comparison.append({
            "Run": label, "MAE": data.mae.mean(), "RMSE": data.rmse.mean(),
            "CRPS": data.crps.mean(), "Brier": data.outbreak_brier.mean(),
            "Precision": data.outbreak_precision.mean(),
            "Specificity": data.outbreak_specificity.mean(),
            "Sensitivity": data.outbreak_sensitivity.mean(), "F1": data.outbreak_f1.mean(),
        })
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(args.output / "previous_vs_retrained_summary.csv", index=False)
    methods = pd.read_csv(reports / "test_selected_method_metrics.csv")
    method_summary = methods.groupby("method", as_index=False)[
        ["precision", "specificity", "sensitivity", "balanced_accuracy", "f1"]
    ].mean()
    method_summary["method"] = method_summary["method"].map(LABELS)
    method_summary = method_summary.sort_values("f1", ascending=False)
    cv = pd.read_csv(reports / "cv_summary_by_horizon_method.csv")
    cv["method"] = cv["method"].map(LABELS)
    cv_display = cv[["horizon", "method", "folds", "mean_f1", "sd_f1", "mean_brier"]]
    policy = pd.read_csv(reports / "selected_incidence_cutoffs.csv")
    selection = json.loads((final / "selection.json").read_text())

    document = Document(); document.styles["Normal"].font.name = "Calibri"
    document.styles["Normal"].font.size = Pt(10.5)
    title = document.add_paragraph(); title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("DEEPST-INLA RETRAIN AND ROLLING TIME-SERIES VALIDATION")
    run.bold = True; run.font.size = Pt(17)
    document.add_heading("1. Protocol", level=1)
    document.add_paragraph(
        "The existing model configuration was retrained without adding an architecture or comparator. "
        "Training was 2010–2018. Validation was a fixed-training, rolling issue-date backtest covering "
        "24 target months in 2019–2020, matching the convention used by dengue_forecast_v1. After policy "
        "selection, the model was refitted on 2010–2020 and evaluated once on the locked 2021–2025 test."
    )
    document.add_paragraph(
        "The panel remains at 554 communes because fcode 31495 (Xa Phien) still has an invalid population "
        "of 47–56 in MongoDB. No population was imputed and no test row was used for selection."
    )
    document.add_heading("2. What cross-validation selected", level=1)
    document.add_paragraph(
        "The rolling validation forecasts were used to select the existing distribution policy, calibrate "
        "the outbreak probability, set its horizon-specific decision cutoff, and select incidence cutoffs "
        "for the v1 consensus rules. Because this is one operational pipeline, no inverse-Brier weighting "
        "across newly introduced models was performed."
    )
    selection_rows = []
    for horizon, values in selection["policies"].items():
        distribution = values["distribution"]
        alert = values["alert"]
        selection_rows.append({
            "H": int(horizon), "distribution": distribution["kind"],
            "decision cutoff": alert["decision_threshold"],
            "validation Brier": alert["brier"], "validation F1": alert["f1"],
        })
    add_table(document, pd.DataFrame(selection_rows))
    document.add_heading("3. Rolling validation by horizon", level=1)
    add_table(document, cv_display)
    document.add_heading("4. Locked-test outbreak methods", level=1)
    add_table(document, method_summary)
    best = method_summary.iloc[0]
    document.add_paragraph(
        f"The best mean H1–H3 test F1 is the current calibrated-probability rule "
        f"({best['f1']:.3f}). The all-three rule is conservative: it improves precision/specificity "
        "but loses substantial sensitivity. The forecast-mean rule and 2-of-3 rule do not improve F1."
    )
    document.add_picture(str(reports / "outbreak_method_comparison.png"), width=Inches(6.4))
    document.add_heading("5. Retrain reproducibility check", level=1)
    add_table(document, comparison)
    document.add_paragraph(
        "The fresh retrain produces very similar aggregate alert performance to the saved run. Small "
        "differences are expected from posterior sampling and validation-selected mixture policies."
    )
    document.add_heading("6. Outputs", level=1)
    document.add_paragraph(
        "Detailed target-level flags, full validation grids, locked-test grids, selected cutoffs, and "
        "fold-level CV metrics are stored beside this report in the outbreak_reports directory."
    )
    document.save(args.output / "Retrained_DeepST_INLA_CV_Report.docx")


if __name__ == "__main__":
    main()
