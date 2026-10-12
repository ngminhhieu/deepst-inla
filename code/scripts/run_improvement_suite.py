#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
DEFAULT_PYTHON = Path(sys.executable)
RSCRIPT_CANDIDATES = [
    Path("/usr/bin/Rscript"),
]


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{stamp()}] {message}", flush=True)


def write_status(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def complete_metrics(path: Path, expected_horizons: int = 3) -> bool:
    metrics = path / "metrics.csv"
    predictions = path / "predictions.csv"
    if not metrics.is_file() or not predictions.is_file():
        return False
    if metrics.stat().st_size <= 0 or predictions.stat().st_size <= 0:
        return False
    return len(metrics.read_text().strip().splitlines()) >= expected_horizons + 1


def run(command: list[str | Path], *, env: dict[str, str], cwd: Path) -> None:
    log("RUN " + " ".join(str(part) for part in command))
    subprocess.run([str(part) for part in command], check=True, cwd=str(cwd), env=env)


def probe_inla(rscript: Path, env: dict[str, str]) -> bool:
    marker = ROOT / "logs" / "inla_probe_marker.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        marker.unlink()
    completed = subprocess.run(
        [
            str(rscript),
            "-e",
            (
                "ok <- requireNamespace('INLA', quietly=TRUE); "
                f"writeLines(ifelse(ok, 'INLA_OK', 'INLA_MISSING'), '{marker}'); "
                "flush.console(); cat(ifelse(ok, 'INLA_OK', 'INLA_MISSING'))"
            ),
        ],
        env=env,
        cwd=str(CODE),
        capture_output=True,
        text=True,
    )
    output = f"{completed.stdout or ''}{completed.stderr or ''}"
    if marker.is_file() and "INLA_OK" in marker.read_text():
        return True
    return "INLA_OK" in output


def resolve_rscript(env: dict[str, str]) -> Path:
    requested = Path(env.get("DENGUE_RSCRIPT", RSCRIPT_CANDIDATES[0]))
    candidates = [requested, *RSCRIPT_CANDIDATES]
    seen = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        if probe_inla(path, env):
            log(f"INLA available via {path}")
            return path
        log(f"INLA probe failed for {path}")
    return requested


def run_deep(
    *,
    python_bin: Path,
    config: Path,
    results_root: Path,
    label: str,
    draws: int,
    env: dict[str, str],
    extra: list[str] | None = None,
) -> None:
    target = results_root / label
    if complete_metrics(target):
        log(f"SKIP completed {label}")
        return
    command = [
        python_bin,
        CODE / "scripts/run_experiment.py",
        "--config",
        config,
        "--model",
        "GNN",
        "--output-name",
        label,
        "--results-root",
        results_root,
        "--draws",
        str(draws),
    ]
    if extra:
        command.extend(extra)
    run(command, env=env, cwd=CODE)


def make_figure(
    *,
    python_bin: Path,
    results_root: Path,
    output: Path,
    models: list[str],
    labels: list[str],
    title: str,
    env: dict[str, str],
) -> None:
    missing = [
        results_root / spec.split("=", 1)[-1]
        for spec in models
        if not (results_root / spec.split("=", 1)[-1]).is_file()
        and not (results_root / spec / "predictions.csv").is_file()
    ]
    resolved_missing = []
    for spec in models:
        if "=" in spec:
            path = results_root / spec.split("=", 1)[1]
        else:
            path = results_root / spec / "predictions.csv"
        if not path.is_file():
            resolved_missing.append(str(path))
    if resolved_missing:
        log("SKIP figure; missing " + ", ".join(resolved_missing))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            python_bin,
            CODE / "scripts/make_comparison_h3_outbreak_figure.py",
            "--results-root",
            results_root,
            "--output",
            output,
            "--models",
            *models,
            "--labels",
            *labels,
            "--title",
            title,
        ],
        env=env,
        cwd=CODE,
    )


def compare_and_lead(
    *,
    python_bin: Path,
    results_root: Path,
    env: dict[str, str],
    models: list[str] | None = None,
) -> None:
    command = [
        python_bin,
        CODE / "scripts/compare_model_metrics.py",
        "--results-root",
        results_root,
        "--output",
        results_root / "model_comparison.csv",
    ]
    if models:
        command.extend(["--models", *models])
    run(command, env=env, cwd=CODE)
    prediction_paths = sorted(
        path for path in results_root.glob("*/predictions.csv") if path.is_file()
    )
    if models:
        prediction_paths = [
            results_root / model / "predictions.csv"
            if not model.endswith("predictions.csv")
            else results_root / model
            for model in models
        ]
        prediction_paths = [
            path if path.name == "predictions.csv" else path / "predictions.csv"
            for path in prediction_paths
        ]
        prediction_paths = [path for path in prediction_paths if path.is_file()]
    if prediction_paths:
        run(
            [
                python_bin,
                CODE / "scripts/evaluate_warning_lead_time.py",
                *prediction_paths,
                "--output",
                results_root / "warning_lead_time.csv",
            ],
            env=env,
            cwd=CODE,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CODE / "config/qc_lognormal.yaml")
    parser.add_argument(
        "--research-root",
        type=Path,
        default=ROOT / "results/research_suite",
    )
    parser.add_argument(
        "--comparison-root",
        type=Path,
        default=ROOT / "results/inla_deep_comparison",
    )
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--skip-inla", action="store_true")
    args = parser.parse_args()

    python_bin = Path(os.environ.get("DENGUE_PYTHON", str(DEFAULT_PYTHON)))
    status_path = ROOT / "logs/improvement_suite_status.txt"
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
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
    inla_ok = probe_inla(rscript, env) and not args.skip_inla
    env["DENGUE_RUN_INLA"] = "1" if inla_ok else "0"
    log(f"python={python_bin}")
    log(f"rscript={rscript}")
    log(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    log(f"INLA={'yes' if inla_ok else 'no'}")

    paper_figure = ROOT / "paper/figures/Fig_H3_predicted_outbreak_examples.png"
    backup = ROOT / "paper/figures/Fig_H3_predicted_outbreak_examples_paper_backup.png"
    if paper_figure.is_file() and not backup.is_file():
        shutil.copy2(paper_figure, backup)
        log(f"backed up existing paper figure to {backup}")

    try:
        subprocess.run(
            [str(python_bin), "-c", "import zstandard"],
            check=False,
            capture_output=True,
        )
        if subprocess.run(
            [str(python_bin), "-c", "import zstandard"],
            capture_output=True,
        ).returncode != 0:
            log("installing zstandard")
            subprocess.run(
                [str(python_bin), "-m", "pip", "install", "--quiet", "zstandard"],
                check=True,
            )

        enriched = CODE / "data/processed/feature_suite/monthly_commune_enriched.csv"
        monthly = CODE / "data/processed/qc_v1/monthly_commune.csv"
        centroids = CODE / "data/processed/qc_v1/commune_centroids.csv"
        adjacency = CODE / "data/processed/qc_v1/administrative_adjacency.csv"
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

        serotype_path = CODE / "data/processed/feature_suite/serotype_province_month.csv"
        serotype_ok = False
        try:
            run(
                [
                    python_bin,
                    CODE / "scripts/fetch_nextstrain_serotype.py",
                    "--monthly",
                    monthly,
                    "--output",
                    serotype_path,
                    "--samples-output",
                    CODE / "data/processed/feature_suite/nextstrain_vietnam_samples.csv",
                ],
                env=env,
                cwd=CODE,
            )
            panel_with_serotype = CODE / "data/processed/feature_suite/monthly_commune_serotype.csv"
            run(
                [
                    python_bin,
                    CODE / "scripts/merge_external_features.py",
                    "--monthly",
                    enriched,
                    "--features",
                    serotype_path,
                    "--level",
                    "province-monthly",
                    "--columns",
                    "denv1_share",
                    "denv2_share",
                    "denv3_share",
                    "denv4_share",
                    "typed_samples_window",
                    "serotype_feature_available",
                    "denv2_dominant",
                    "--output",
                    panel_with_serotype,
                ],
                env=env,
                cwd=CODE,
            )
            serotype_ok = True
        except subprocess.CalledProcessError as exc:
            log(f"SKIP serotype features: {exc}")
            panel_with_serotype = enriched

        make_figure(
            python_bin=python_bin,
            results_root=args.research_root,
            output=ROOT / "results/refactor_summary/Fig_H3_predicted_outbreak_examples",
            models=["DEEP-BASE-LN", "DEEP-BASE-NB", "DEEP-MOBILITY-GRAVITY"],
            labels=["Deep LN baseline", "Deep NB", "Gravity mobility"],
            title="H3 predicted outbreak examples: current deep research models",
            env=env,
        )
        make_figure(
            python_bin=python_bin,
            results_root=args.research_root,
            output=ROOT / "paper/figures/Fig_H3_predicted_outbreak_examples",
            models=["DEEP-BASE-LN", "DEEP-BASE-NB", "DEEP-MOBILITY-GRAVITY"],
            labels=["Deep LN baseline", "Deep NB", "Gravity mobility"],
            title="H3 predicted outbreak examples: current deep research models",
            env=env,
        )

        climate_extra = [
            "--monthly-path",
            str(enriched),
            "--climate-columns",
            "t2m_avg",
            "rh_avg",
            "tp_accum",
            "t2m_anomaly",
            "tp_anomaly",
            "rh_anomaly",
        ]
        jobs = [
            (
                "DEEP-NEIGHBOR",
                [
                    "--monthly-path",
                    str(enriched),
                    "--history-covariate-columns",
                    "neighbor_mean_log_incidence",
                    "neighbor_max_log_incidence",
                ],
            ),
            (
                "DEEP-CLIMATE-ANOMALY",
                climate_extra,
            ),
            (
                "DEEP-PROVINCE-CONTEXT",
                [
                    "--monthly-path",
                    str(enriched),
                    "--history-covariate-columns",
                    "province_loo_log_incidence",
                ],
            ),
            (
                "DEEP-EW-SOFT",
                [
                    "--outbreak-loss-weight",
                    "0.05",
                    "--alert-probability-source",
                    "auxiliary",
                ],
            ),
            (
                "DEEP-NB-FEATURES",
                [
                    "--likelihood",
                    "negative_binomial",
                    "--monthly-path",
                    str(enriched),
                    "--history-covariate-columns",
                    "gravity_import_pressure",
                    "log_population_density",
                    "neighbor_mean_log_incidence",
                    "case_growth_1m",
                    "case_growth_3m",
                ],
            ),
        ]
        if serotype_ok:
            jobs.extend(
                [
                    (
                        "DEEP-SEROTYPE",
                        [
                            "--monthly-path",
                            str(panel_with_serotype),
                            "--history-covariate-columns",
                            "denv1_share",
                            "denv2_share",
                            "denv3_share",
                            "denv4_share",
                            "serotype_feature_available",
                            "denv2_dominant",
                        ],
                    ),
                    (
                        "DEEP-NB-SEROTYPE-SPATIAL",
                        [
                            "--likelihood",
                            "negative_binomial",
                            "--monthly-path",
                            str(panel_with_serotype),
                            "--history-covariate-columns",
                            "gravity_import_pressure",
                            "neighbor_mean_log_incidence",
                            "denv2_share",
                            "denv2_dominant",
                            "serotype_feature_available",
                            "case_growth_1m",
                        ],
                    ),
                ]
            )

        for label, extra in jobs:
            run_deep(
                python_bin=python_bin,
                config=args.config,
                results_root=args.research_root,
                label=label,
                draws=args.draws,
                env=env,
                extra=extra,
            )

        ensemble = args.research_root / "ENSEMBLE-NB-MOBILITY-ALL"
        members = [
            args.research_root / "DEEP-BASE-NB",
            args.research_root / "DEEP-MOBILITY-GRAVITY",
            args.research_root / "DEEP-ALL-AVAILABLE",
        ]
        if all(complete_metrics(path) for path in members) and not complete_metrics(ensemble):
            run(
                [
                    python_bin,
                    CODE / "scripts/ensemble_model_predictions.py",
                    "--output",
                    ensemble,
                    "--model-label",
                    "ENSEMBLE-NB-MOBILITY-ALL",
                    "--members",
                    *members,
                ],
                env=env,
                cwd=CODE,
            )
        elif complete_metrics(ensemble):
            log("SKIP completed ENSEMBLE-NB-MOBILITY-ALL")

        compare_and_lead(
            python_bin=python_bin,
            results_root=args.research_root,
            env=env,
        )
        summary_root = ROOT / "results/refactor_summary"
        summary_root.mkdir(parents=True, exist_ok=True)
        if (args.research_root / "model_comparison.csv").is_file():
            shutil.copy2(
                args.research_root / "model_comparison.csv",
                summary_root / "research_model_comparison.csv",
            )

        research_figure_models = [
            "DEEP-BASE-LN",
            "DEEP-BASE-NB",
            "ENSEMBLE-NB-MOBILITY-ALL",
        ]
        if complete_metrics(args.research_root / "DEEP-SEROTYPE"):
            research_figure_models = [
                "DEEP-BASE-LN",
                "DEEP-SEROTYPE",
                "ENSEMBLE-NB-MOBILITY-ALL",
            ]
        elif complete_metrics(args.research_root / "DEEP-NB-FEATURES"):
            research_figure_models = [
                "DEEP-BASE-LN",
                "DEEP-NB-FEATURES",
                "ENSEMBLE-NB-MOBILITY-ALL",
            ]
        make_figure(
            python_bin=python_bin,
            results_root=args.research_root,
            output=ROOT / "paper/figures/Fig_H3_predicted_outbreak_examples",
            models=research_figure_models,
            labels=[name.replace("-", " ") for name in research_figure_models],
            title="H3 predicted outbreak examples from the improved research suite",
            env=env,
        )
        shutil.copy2(
            ROOT / "paper/figures/Fig_H3_predicted_outbreak_examples.png",
            summary_root / "Fig_H3_predicted_outbreak_examples.png",
        )

        if inla_ok:
            from run_refactor_model_suite import run_comparison

            run_comparison(
                python_bin=python_bin,
                rscript=rscript,
                config=args.config,
                results_root=args.comparison_root,
                draws=args.draws,
                env=env,
                paths={
                    "monthly_path": monthly.resolve(),
                    "adjacency_path": adjacency.resolve(),
                    "centroids_path": centroids.resolve(),
                },
            )
            if complete_metrics(args.comparison_root / "DEEP-ONLY") and complete_metrics(
                args.comparison_root / "INLA-ONLY" / "INLA-TS"
            ) and complete_metrics(args.comparison_root / "DEEP-INLA" / "combined"):
                make_figure(
                    python_bin=python_bin,
                    results_root=args.comparison_root,
                    output=ROOT / "paper/figures/Fig_H3_deep_inla_outbreak_examples",
                    models=[
                        "DEEP-ONLY=DEEP-ONLY/predictions.csv",
                        "INLA-TS=INLA-ONLY/INLA-TS/predictions.csv",
                        "DEEP-INLA=DEEP-INLA/combined/predictions.csv",
                    ],
                    labels=["Deep-only GNN", "INLA-TS", "Deep + INLA"],
                    title="H3 predicted outbreak examples: deep-only vs INLA-only vs deep+INLA",
                    env=env,
                )
            if (args.comparison_root / "model_comparison.csv").is_file():
                shutil.copy2(
                    args.comparison_root / "model_comparison.csv",
                    summary_root / "core_model_comparison.csv",
                )
        else:
            log("SKIP INLA comparison")

        write_status(status_path, f"COMPLETE finished={stamp()}")
        log("COMPLETE improvement suite")
    except Exception as exc:
        write_status(status_path, f"FAILED finished={stamp()} error={exc}")
        log(f"FAILED {exc}")
        raise


if __name__ == "__main__":
    try:
        sys.path.insert(0, str(CODE / "scripts"))
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
