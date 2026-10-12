#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def run_inla_tolerant(
    command: list[str | Path],
    *,
    env: dict[str, str],
    cwd: Path,
    output: Path,
) -> None:
    try:
        run(command, env=env, cwd=cwd)
    except subprocess.CalledProcessError as exc:
        if complete_metrics(output):
            log(f"INLA aborted after writing metrics at {output}; continuing")
            return
        raise exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CODE / "config/qc_lognormal.yaml")
    parser.add_argument("--research-root", type=Path, default=ROOT / "results/research_suite")
    parser.add_argument(
        "--comparison-root",
        type=Path,
        default=ROOT / "results/inla_deep_comparison",
    )
    parser.add_argument("--draws", type=int, default=500)
    args = parser.parse_args()

    python_bin = Path(os.environ.get("DENGUE_PYTHON", str(DEFAULT_PYTHON)))
    status_path = ROOT / "logs/open_data_suite_status.txt"
    env = os.environ.copy()
    env.update(
        {
            "DENGUE_PYTHON": str(python_bin),
            "PYTHONPATH": f"{CODE}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep),
            "R_LIBS_USER": str(ROOT / ".r-library"),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0"),
            "MPLBACKEND": "Agg",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "4"),
            "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS", "4"),
        }
    )
    write_status(status_path, f"RUNNING started={stamp()}")
    rscript = resolve_rscript(env)
    env["DENGUE_RSCRIPT"] = str(rscript)
    env["PATH"] = f"{rscript.parent}{os.pathsep}{env.get('PATH', '')}"
    log(f"python={python_bin} rscript={rscript} gpu={env['CUDA_VISIBLE_DEVICES']}")

    monthly = CODE / "data/processed/qc_v1/monthly_commune.csv"
    centroids = CODE / "data/processed/qc_v1/commune_centroids.csv"
    adjacency = CODE / "data/processed/qc_v1/administrative_adjacency.csv"
    enriched = CODE / "data/processed/feature_suite/monthly_commune_enriched.csv"
    open_panel = CODE / "data/processed/feature_suite/monthly_commune_open.csv"
    future_path = CODE / "data/processed/feature_suite/future_climate_climatology.csv"

    try:
        run(
            [
                python_bin,
                CODE / "scripts/build_feature_panel.py",
                "--monthly",
                monthly,
                "--centroids",
                centroids,
                "--adjacency",
                adjacency,
                "--output",
                enriched,
            ],
            env=env,
            cwd=CODE,
        )
        run(
            [
                python_bin,
                CODE / "scripts/build_open_covariates.py",
                "--monthly",
                enriched,
                "--centroids",
                centroids,
                "--output-monthly",
                open_panel,
                "--output-future",
                future_path,
                "--cache-dir",
                CODE / "data/external",
            ],
            env=env,
            cwd=CODE,
        )
        available = set(pd_columns(open_panel))

        def history(label: str, columns: list[str], extra: list[str] | None = None) -> None:
            present = [column for column in columns if column in available]
            if not present:
                log(f"SKIP {label}: missing {columns}")
                return
            command = ["--monthly-path", str(open_panel), "--history-covariate-columns", *present]
            if extra:
                command.extend(extra)
            run_deep(
                python_bin=python_bin,
                config=args.config,
                results_root=args.research_root,
                label=label,
                draws=args.draws,
                env=env,
                extra=command,
            )

        land_climate = [
            column
            for column in ("soil_moisture", "solar_downward", "dewpoint_c", "et0_mm")
            if column in available
        ]
        if land_climate:
            run_deep(
                python_bin=python_bin,
                config=args.config,
                results_root=args.research_root,
                label="DEEP-LAND-SURFACE",
                draws=args.draws,
                env=env,
                extra=[
                    "--monthly-path",
                    str(open_panel),
                    "--climate-columns",
                    "t2m_avg",
                    "rh_avg",
                    "tp_accum",
                    *land_climate,
                ],
            )
        history("DEEP-ENSO", ["nino34_lag2", "iod_dmi_lag1", "mosquito_suitability"])
        history("DEEP-RADIATION-MOBILITY", ["radiation_import_pressure"])
        history("DEEP-ELEVATION", ["elevation_m"])
        history("DEEP-BUILTUP", ["log_builtup"])
        history("DEEP-MEKONG-NEIGHBORS", ["neighbor_dengue_log"])
        history(
            "DEEP-OPEN-COMBO",
            [
                "radiation_import_pressure",
                "nino34_lag2",
                "iod_dmi_lag1",
                "mosquito_suitability",
                "elevation_m",
                "log_builtup",
                "neighbor_dengue_log",
                "soil_moisture",
            ],
        )
        history(
            "DEEP-NB-OPEN",
            [
                "radiation_import_pressure",
                "nino34_lag2",
                "mosquito_suitability",
                "neighbor_dengue_log",
                "gravity_import_pressure",
            ],
            extra=["--likelihood", "negative_binomial"],
        )
        if future_path.is_file():
            run_deep(
                python_bin=python_bin,
                config=args.config,
                results_root=args.research_root,
                label="DEEP-FUTURE-CLIMATE",
                draws=args.draws,
                env=env,
                extra=[
                    "--future-covariates-path",
                    str(future_path),
                    "--future-covariate-columns",
                    "t2m_forecast",
                    "rh_forecast",
                    "tp_forecast",
                ],
            )

        ensemble_members = [
            args.research_root / "DEEP-BASE-NB",
            args.research_root / "DEEP-NB-FEATURES",
            args.research_root / "DEEP-OPEN-COMBO",
        ]
        if all(complete_metrics(path) for path in ensemble_members):
            ensemble = args.research_root / "ENSEMBLE-NB-OPEN"
            if not complete_metrics(ensemble):
                run(
                    [
                        python_bin,
                        CODE / "scripts/ensemble_model_predictions.py",
                        "--output",
                        ensemble,
                        "--model-label",
                        "ENSEMBLE-NB-OPEN",
                        "--members",
                        *ensemble_members,
                    ],
                    env=env,
                    cwd=CODE,
                )

        run(
            [
                python_bin,
                CODE / "scripts/compare_model_metrics.py",
                "--results-root",
                args.research_root,
                "--output",
                args.research_root / "model_comparison.csv",
            ],
            env=env,
            cwd=CODE,
        )
        prediction_paths = sorted(args.research_root.glob("*/predictions.csv"))
        if prediction_paths:
            run(
                [
                    python_bin,
                    CODE / "scripts/evaluate_warning_lead_time.py",
                    *prediction_paths,
                    "--output",
                    args.research_root / "warning_lead_time.csv",
                ],
                env=env,
                cwd=CODE,
            )
        shutil.copy2(
            args.research_root / "model_comparison.csv",
            ROOT / "results/refactor_summary/research_model_comparison.csv",
        )

        figure_models = ["DEEP-BASE-NB", "DEEP-NB-FEATURES", "DEEP-FUTURE-CLIMATE"]
        if complete_metrics(args.research_root / "ENSEMBLE-NB-OPEN"):
            figure_models[2] = "ENSEMBLE-NB-OPEN"
        elif complete_metrics(args.research_root / "DEEP-OPEN-COMBO"):
            figure_models[2] = "DEEP-OPEN-COMBO"
        elif complete_metrics(args.research_root / "DEEP-NB-OPEN"):
            figure_models[2] = "DEEP-NB-OPEN"
        figure_models = [name for name in figure_models if complete_metrics(args.research_root / name)]
        if len(figure_models) >= 2:
            make_figure(
                python_bin=python_bin,
                results_root=args.research_root,
                output=ROOT / "paper/figures/Fig_H3_predicted_outbreak_examples",
                models=figure_models[:3],
                labels=[name.replace("-", " ") for name in figure_models[:3]],
                title="H3 predicted outbreak examples with open environmental and mobility data",
                env=env,
            )
            shutil.copy2(
                ROOT / "paper/figures/Fig_H3_predicted_outbreak_examples.png",
                ROOT / "results/refactor_summary/Fig_H3_predicted_outbreak_examples.png",
            )

        inla_root = args.comparison_root / "INLA-ONLY"
        inla_ts = inla_root / "INLA-TS"
        rows_cache = args.comparison_root / "inla_rows_cache.rds"
        if complete_metrics(inla_ts):
            log("SKIP completed INLA-ONLY/INLA-TS")
        else:
            inla_root.mkdir(parents=True, exist_ok=True)
            local_env = env.copy()
            local_env.update(
                {
                    "DENGUE_EVAL_SPLIT": "test",
                    "DENGUE_STATIC_MODELS": "INLA-TS",
                    "DENGUE_SKIP_STATIC_INLA": "false",
                    "DENGUE_RUN_STRONG_INLA": "false",
                    "DENGUE_SAVE_SAMPLES": "true",
                    "DENGUE_ROWS_CACHE": str(rows_cache),
                }
            )
            run_inla_tolerant(
                [
                    rscript,
                    CODE / "r/inla_baselines.R",
                    monthly,
                    adjacency,
                    inla_root,
                    str(args.draws),
                    "1234",
                ],
                env=local_env,
                cwd=CODE,
                output=inla_ts,
            )

        fusion_root = args.comparison_root / "DEEP-INLA"
        combined = fusion_root / "combined"
        deep_only = args.comparison_root / "DEEP-ONLY"
        if complete_metrics(combined):
            log("SKIP completed DEEP-INLA/combined")
        elif complete_metrics(deep_only) and (deep_only / "representations.csv").is_file() and rows_cache.is_file():
            for horizon in (1, 2, 3):
                output = fusion_root / f"H{horizon}"
                if complete_metrics(output, expected_horizons=1):
                    log(f"SKIP completed DEEP-INLA/H{horizon}")
                    continue
                output.mkdir(parents=True, exist_ok=True)
                run_inla_tolerant(
                    [
                        rscript,
                        CODE / "r/representation_augmented_inla.R",
                        rows_cache,
                        adjacency,
                        deep_only / "representations.csv",
                        output,
                        str(horizon),
                        str(args.draws),
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
                    output=output,
                )
            if all(complete_metrics(fusion_root / f"H{horizon}", expected_horizons=1) for horizon in (1, 2, 3)):
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
        else:
            log("SKIP DEEP-INLA: missing DEEP-ONLY representations or INLA row cache")

        if complete_metrics(deep_only) and complete_metrics(inla_ts):
            models = ["DEEP-ONLY=DEEP-ONLY/predictions.csv", "INLA-TS=INLA-ONLY/INLA-TS/predictions.csv"]
            labels = ["Deep-only GNN", "INLA-TS"]
            if complete_metrics(combined):
                models.append("DEEP-INLA=DEEP-INLA/combined/predictions.csv")
                labels.append("Deep + INLA")
            make_figure(
                python_bin=python_bin,
                results_root=args.comparison_root,
                output=ROOT / "paper/figures/Fig_H3_deep_inla_outbreak_examples",
                models=models,
                labels=labels,
                title="H3 predicted outbreak examples: deep-only vs INLA-only vs deep+INLA",
                env=env,
            )
            run(
                [
                    python_bin,
                    CODE / "scripts/compare_model_metrics.py",
                    "--results-root",
                    args.comparison_root,
                    "--models",
                    "DEEP-ONLY",
                    "INLA-ONLY/INLA-TS",
                    *(["DEEP-INLA/combined"] if complete_metrics(combined) else []),
                    "--output",
                    args.comparison_root / "model_comparison.csv",
                ],
                env=env,
                cwd=CODE,
            )
            shutil.copy2(
                args.comparison_root / "model_comparison.csv",
                ROOT / "results/refactor_summary/core_model_comparison.csv",
            )

        write_status(status_path, f"COMPLETE finished={stamp()}")
        log("COMPLETE open-data suite")
    except Exception as exc:
        write_status(status_path, f"FAILED finished={stamp()} error={exc}")
        log(f"FAILED {exc}")
        raise


def pd_columns(path: Path) -> list[str]:
    import pandas as pd

    return list(pd.read_csv(path, nrows=0).columns)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
