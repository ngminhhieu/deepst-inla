#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from run_improvement_suite import (
    CODE,
    DEFAULT_PYTHON,
    ROOT,
    complete_metrics,
    log,
    make_figure,
    resolve_rscript,
    run,
    run_deep,
    stamp,
    write_status,
)


def main() -> None:
    python_bin = Path(os.environ.get("DENGUE_PYTHON", str(DEFAULT_PYTHON)))
    research_root = ROOT / "results/research_suite"
    comparison_root = ROOT / "results/inla_deep_comparison"
    config = CODE / "config/qc_lognormal.yaml"
    open_panel = CODE / "data/processed/feature_suite/monthly_commune_open.csv"
    status_path = ROOT / "logs/survey_improvement_status.txt"
    env = os.environ.copy()
    env.update(
        {
            "DENGUE_PYTHON": str(python_bin),
            "PYTHONPATH": f"{CODE}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(
                os.pathsep
            ),
            "R_LIBS_USER": str(ROOT / ".r-library"),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0"),
            "MPLBACKEND": "Agg",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        }
    )
    write_status(status_path, f"RUNNING started={stamp()}")
    try:
        jobs = [
            (
                "VAL-NB-BASE",
                [
                    "--likelihood",
                    "negative_binomial",
                    "--export-validation-predictions",
                ],
            ),
            (
                "VAL-NB-PEAK",
                [
                    "--likelihood",
                    "negative_binomial",
                    "--peak-loss-weight",
                    "2.0",
                    "--export-validation-predictions",
                ],
            ),
            (
                "VAL-LN-LAND",
                [
                    "--monthly-path",
                    str(open_panel),
                    "--climate-columns",
                    "t2m_avg",
                    "rh_avg",
                    "tp_accum",
                    "soil_moisture",
                    "solar_downward",
                    "dewpoint_c",
                    "et0_mm",
                    "--export-validation-predictions",
                ],
            ),
            (
                "VAL-EW-FOCAL",
                [
                    "--outbreak-loss-weight",
                    "0.10",
                    "--outbreak-positive-weight",
                    "auto",
                    "--outbreak-focal-gamma",
                    "2.0",
                    "--alert-probability-source",
                    "auxiliary",
                    "--export-validation-predictions",
                ],
            ),
        ]
        for label, extra in jobs:
            run_deep(
                python_bin=python_bin,
                config=config,
                results_root=research_root,
                label=label,
                draws=500,
                env=env,
                extra=extra,
            )
            tuned = research_root / f"{label}-TUNED"
            if not complete_metrics(tuned):
                run(
                    [
                        python_bin,
                        CODE / "scripts/tune_alert_threshold.py",
                        research_root / label,
                        tuned,
                        "--model-label",
                        f"{label}-TUNED",
                        "--minimum-precision",
                        "0.50",
                    ],
                    env=env,
                    cwd=CODE,
                )

        stacked = research_root / "STACK-VALIDATED"
        if not complete_metrics(stacked):
            run(
                [
                    python_bin,
                    CODE / "scripts/stack_predictive_distributions.py",
                    "--output",
                    stacked,
                    "--model-label",
                    "STACK-VALIDATED",
                    "--members",
                    research_root / "VAL-NB-BASE",
                    research_root / "VAL-NB-PEAK",
                    research_root / "VAL-LN-LAND",
                    "--selection-draws",
                    "100",
                    "--crps-tolerance",
                    "0.03",
                    "--minimum-precision",
                    "0.50",
                ],
                env=env,
                cwd=CODE,
            )

        run(
            [
                python_bin,
                CODE / "scripts/compare_model_metrics.py",
                "--results-root",
                research_root,
                "--output",
                research_root / "model_comparison.csv",
            ],
            env=env,
            cwd=CODE,
        )
        prediction_paths = sorted(research_root.glob("*/predictions.csv"))
        run(
            [
                python_bin,
                CODE / "scripts/evaluate_warning_lead_time.py",
                *prediction_paths,
                "--output",
                research_root / "warning_lead_time.csv",
            ],
            env=env,
            cwd=CODE,
        )
        shutil.copy2(
            research_root / "model_comparison.csv",
            ROOT / "results/refactor_summary/research_model_comparison.csv",
        )
        make_figure(
            python_bin=python_bin,
            results_root=research_root,
            output=ROOT / "paper/figures/Fig_H3_predicted_outbreak_examples",
            models=[
                "VAL-NB-BASE-TUNED",
                "VAL-EW-FOCAL-TUNED",
                "STACK-VALIDATED",
            ],
            labels=[
                "NB + validation alert threshold",
                "Class-balanced focal early warning",
                "Validation-stacked forecast",
            ],
            title="H3 outbreak forecasts after validation-based calibration",
            env=env,
        )

        rscript = resolve_rscript(env)
        env["DENGUE_RSCRIPT"] = str(rscript)
        fusion_root = comparison_root / "DEEP-INLA"
        rows_cache = comparison_root / "inla_rows_cache.rds"
        representations = comparison_root / "DEEP-ONLY/representations.csv"
        adjacency = CODE / "data/processed/qc_v1/administrative_adjacency.csv"
        for horizon in (1, 2, 3):
            output = fusion_root / f"H{horizon}"
            if complete_metrics(output, expected_horizons=1):
                log(f"SKIP completed DEEP-INLA/H{horizon}")
                continue
            output.mkdir(parents=True, exist_ok=True)
            try:
                run(
                    [
                        rscript,
                        CODE / "r/representation_augmented_inla.R",
                        rows_cache,
                        adjacency,
                        representations,
                        output,
                        str(horizon),
                        "500",
                        "1234",
                        "DEEP-INLA",
                        "latent_residual_pca",
                        "4",
                        "5",
                        "1",
                        "test",
                    ],
                    env=env,
                    cwd=CODE,
                )
            except subprocess.CalledProcessError:
                if not complete_metrics(output, expected_horizons=1):
                    raise
                log(f"DEEP-INLA/H{horizon} aborted after writing complete output")
        combined = fusion_root / "combined"
        if not complete_metrics(combined):
            run(
                [
                    python_bin,
                    CODE / "scripts/combine_augmented_inla_horizons.py",
                    combined,
                    fusion_root / "H1",
                    fusion_root / "H2",
                    fusion_root / "H3",
                ],
                env=env,
                cwd=CODE,
            )
        run(
            [
                python_bin,
                CODE / "scripts/compare_model_metrics.py",
                "--results-root",
                comparison_root,
                "--models",
                "DEEP-ONLY",
                "INLA-ONLY/INLA-TS",
                "DEEP-INLA/combined",
                "--output",
                comparison_root / "model_comparison.csv",
            ],
            env=env,
            cwd=CODE,
        )
        shutil.copy2(
            comparison_root / "model_comparison.csv",
            ROOT / "results/refactor_summary/core_model_comparison.csv",
        )
        make_figure(
            python_bin=python_bin,
            results_root=comparison_root,
            output=ROOT / "paper/figures/Fig_H3_deep_inla_outbreak_examples",
            models=[
                "DEEP-ONLY=DEEP-ONLY/predictions.csv",
                "INLA-TS=INLA-ONLY/INLA-TS/predictions.csv",
                "DEEP-INLA=DEEP-INLA/combined/predictions.csv",
            ],
            labels=["Deep-only GNN", "INLA-TS", "Deep + INLA"],
            title="H3 predicted outbreaks: deep-only vs INLA-only vs deep+INLA",
            env=env,
        )
        write_status(status_path, f"COMPLETE finished={stamp()}")
        log("COMPLETE survey improvement suite")
    except Exception as exc:
        write_status(status_path, f"FAILED finished={stamp()} error={exc}")
        log(f"FAILED {exc}")
        raise


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
