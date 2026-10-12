#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
DEFAULT_PYTHON = Path(sys.executable)
DEFAULT_RSCRIPT = Path("/usr/bin/Rscript")


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{stamp()}] {message}", flush=True)


def complete_metrics(path: Path, expected_horizons: int = 3) -> bool:
    metrics = path / "metrics.csv"
    predictions = path / "predictions.csv"
    if not metrics.is_file() or not predictions.is_file():
        return False
    if metrics.stat().st_size <= 0 or predictions.stat().st_size <= 0:
        return False
    lines = metrics.read_text().strip().splitlines()
    return len(lines) >= expected_horizons + 1


def run(command: list[str], *, env: dict[str, str], cwd: Path) -> None:
    log("RUN " + " ".join(str(part) for part in command))
    subprocess.run([str(part) for part in command], check=True, cwd=str(cwd), env=env)


def ensure_inla(env: dict[str, str]) -> bool:
    # Some INLA builds abort during process teardown even after a successful load.
    # Detect availability from stdout instead of relying on process exit codes.
    probe = [
        env["DENGUE_RSCRIPT"],
        "-e",
        "cat(ifelse(requireNamespace('INLA', quietly=TRUE), 'INLA_OK', 'INLA_MISSING'))",
    ]
    completed = subprocess.run(
        probe,
        env=env,
        cwd=str(CODE),
        capture_output=True,
        text=True,
    )
    output = f"{completed.stdout or ''}{completed.stderr or ''}"
    return "INLA_OK" in output


def write_status(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def resolve_data_paths(config_path: Path) -> dict[str, Path]:
    cfg = yaml.safe_load(config_path.read_text())
    root = config_path.resolve().parent.parent
    output = {}
    for key in ("monthly_path", "adjacency_path", "centroids_path"):
        value = Path(cfg["data"][key])
        output[key] = value if value.is_absolute() else (root / value).resolve()
    return output


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


def run_comparison(
    *,
    python_bin: Path,
    rscript: Path,
    config: Path,
    results_root: Path,
    draws: int,
    env: dict[str, str],
    paths: dict[str, Path],
) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    deep_only = results_root / "DEEP-ONLY"
    if complete_metrics(deep_only) and (deep_only / "representations.csv").is_file():
        log("SKIP completed DEEP-ONLY")
    else:
        run_deep(
            python_bin=python_bin,
            config=config,
            results_root=results_root,
            label="DEEP-ONLY",
            draws=draws,
            env=env,
            extra=["--export-representations"],
        )

    if env.get("DENGUE_RUN_INLA") != "1":
        run(
            [
                python_bin,
                CODE / "scripts/compare_model_metrics.py",
                "--results-root",
                results_root,
                "--models",
                "DEEP-ONLY",
                "--output",
                results_root / "model_comparison.csv",
            ],
            env=env,
            cwd=CODE,
        )
        return

    inla_root = results_root / "INLA-ONLY"
    inla_ts = inla_root / "INLA-TS"
    rows_cache = results_root / "inla_rows_cache.rds"
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
        run(
            [
                rscript,
                CODE / "r/inla_baselines.R",
                paths["monthly_path"],
                paths["adjacency_path"],
                inla_root,
                str(draws),
                "1234",
            ],
            env=local_env,
            cwd=CODE,
        )

    fusion_root = results_root / "DEEP-INLA"
    combined = fusion_root / "combined"
    if complete_metrics(combined):
        log("SKIP completed DEEP-INLA/combined")
    else:
        for horizon in (1, 2, 3):
            output = fusion_root / f"H{horizon}"
            if complete_metrics(output, expected_horizons=1):
                log(f"SKIP completed DEEP-INLA/H{horizon}")
                continue
            output.mkdir(parents=True, exist_ok=True)
            run(
                [
                    rscript,
                    CODE / "r/representation_augmented_inla.R",
                    rows_cache,
                    paths["adjacency_path"],
                    deep_only / "representations.csv",
                    output,
                    str(horizon),
                    str(draws),
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
            results_root,
            "--models",
            "DEEP-ONLY",
            "INLA-ONLY/INLA-TS",
            "DEEP-INLA/combined",
            "--output",
            results_root / "model_comparison.csv",
        ],
        env=env,
        cwd=CODE,
    )
    run(
        [
            python_bin,
            CODE / "scripts/evaluate_warning_lead_time.py",
            deep_only / "predictions.csv",
            inla_ts / "predictions.csv",
            combined / "predictions.csv",
            "--output",
            results_root / "warning_lead_time.csv",
        ],
        env=env,
        cwd=CODE,
    )


def run_research(
    *,
    python_bin: Path,
    rscript: Path,
    config: Path,
    results_root: Path,
    draws: int,
    env: dict[str, str],
    paths: dict[str, Path],
) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    enriched = CODE / "data/processed/feature_suite/monthly_commune_enriched.csv"
    enriched.parent.mkdir(parents=True, exist_ok=True)
    if not enriched.is_file():
        run(
            [
                python_bin,
                CODE / "scripts/build_feature_panel.py",
                "--monthly",
                paths["monthly_path"],
                "--centroids",
                paths["centroids_path"],
                "--output",
                enriched,
            ],
            env=env,
            cwd=CODE,
        )
    else:
        log(f"SKIP existing feature panel {enriched}")

    climate_extra = [
        "--monthly-path",
        str(enriched),
        "--climate-columns",
        "t2m_avg",
        "rh_avg",
        "tp_accum",
        "ws_avg",
        "ws_min",
        "ws_max",
        "t2m_min",
        "t2m_max",
        "rh_min",
        "rh_max",
        "rain_3m_sum",
        "rain_6m_sum",
        "temp_3m_mean",
        "humidity_3m_mean",
    ]
    jobs = [
        ("DEEP-BASE-LN", ["--export-representations", "--export-validation-predictions"]),
        ("DEEP-BASE-NB", ["--likelihood", "negative_binomial"]),
        ("DEEP-EXTENDED-CLIMATE", climate_extra),
        (
            "DEEP-MOBILITY-GRAVITY",
            ["--monthly-path", str(enriched), "--history-covariate-columns", "gravity_import_pressure"],
        ),
        (
            "DEEP-DENSITY",
            ["--monthly-path", str(enriched), "--history-covariate-columns", "log_population_density"],
        ),
        (
            "DEEP-EPIDEMIC-GROWTH",
            [
                "--monthly-path",
                str(enriched),
                "--history-covariate-columns",
                "case_growth_1m",
                "case_growth_3m",
            ],
        ),
        (
            "DEEP-ALL-AVAILABLE",
            climate_extra
            + [
                "--history-covariate-columns",
                "gravity_import_pressure",
                "log_population_density",
                "case_growth_1m",
                "case_growth_3m",
            ],
        ),
        (
            "DEEP-EARLY-WARNING",
            [
                "--outbreak-loss-weight",
                env.get("DENGUE_OUTBREAK_LOSS_WEIGHT", "0.25"),
                "--alert-probability-source",
                "auxiliary",
            ],
        ),
    ]
    for label, extra in jobs:
        run_deep(
            python_bin=python_bin,
            config=config,
            results_root=results_root,
            label=label,
            draws=draws,
            env=env,
            extra=extra,
        )

    if env.get("DENGUE_RUN_INLA") == "1":
        deep_inla = results_root / "DEEP-INLA"
        if complete_metrics(deep_inla):
            log("SKIP completed research DEEP-INLA")
        else:
            run(
                [
                    "bash",
                    CODE / "scripts/run_deep_inla_all.sh",
                    results_root / "DEEP-BASE-LN" / "representations.csv",
                    deep_inla,
                    str(draws),
                    "1234",
                    "DEEP-INLA",
                ],
                env=env,
                cwd=CODE,
            )
    else:
        log("SKIP research DEEP-INLA because INLA is unavailable")

    run(
        [
            python_bin,
            CODE / "scripts/compare_model_metrics.py",
            "--results-root",
            results_root,
            "--output",
            results_root / "model_comparison.csv",
        ],
        env=env,
        cwd=CODE,
    )
    prediction_paths = sorted(
        path
        for path in results_root.glob("*/predictions.csv")
        if path.is_file()
    )
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
        "--comparison-root",
        type=Path,
        default=ROOT / "results/inla_deep_comparison",
    )
    parser.add_argument(
        "--research-root",
        type=Path,
        default=ROOT / "results/research_suite",
    )
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--skip-research", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    python_bin = Path(os.environ.get("DENGUE_PYTHON", str(DEFAULT_PYTHON)))
    rscript = Path(os.environ.get("DENGUE_RSCRIPT", str(DEFAULT_RSCRIPT)))
    if not python_bin.is_file():
        raise SystemExit(f"Python not found: {python_bin}")
    if not rscript.is_file():
        rscript = Path("/usr/bin/Rscript")

    status_path = ROOT / "logs/refactor_suite_status.txt"
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    paths = resolve_data_paths(args.config)

    env = os.environ.copy()
    env.update(
        {
            "DENGUE_PYTHON": str(python_bin),
            "DENGUE_RSCRIPT": str(rscript),
            "DENGUE_DRAWS": str(args.draws),
            "DENGUE_RUN_INLA": "1",
            "PYTHONPATH": f"{CODE}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep),
            "R_LIBS_USER": str(ROOT / ".r-library"),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "1"),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "4"),
            "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS", "4"),
            "PATH": f"{rscript.parent}{os.pathsep}{env.get('PATH', '')}",
        }
    )

    write_status(status_path, f"RUNNING started={stamp()}")
    log(f"python={python_bin}")
    log(f"rscript={rscript}")
    log(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")

    if not ensure_inla(env):
        log("WARN INLA unavailable; deep-only paths will still run")
        env["DENGUE_RUN_INLA"] = "0"

    try:
        if not args.skip_comparison:
            run_comparison(
                python_bin=python_bin,
                rscript=rscript,
                config=args.config,
                results_root=args.comparison_root,
                draws=args.draws,
                env=env,
                paths=paths,
            )
        if not args.skip_research:
            run_research(
                python_bin=python_bin,
                rscript=rscript,
                config=args.config,
                results_root=args.research_root,
                draws=args.draws,
                env=env,
                paths=paths,
            )

        summary_root = ROOT / "results/refactor_summary"
        summary_root.mkdir(parents=True, exist_ok=True)
        if (args.comparison_root / "model_comparison.csv").is_file():
            (summary_root / "core_model_comparison.csv").write_bytes(
                (args.comparison_root / "model_comparison.csv").read_bytes()
            )
        if (args.research_root / "model_comparison.csv").is_file():
            (summary_root / "research_model_comparison.csv").write_bytes(
                (args.research_root / "model_comparison.csv").read_bytes()
            )

        if not args.skip_figures and complete_metrics(
            args.comparison_root / "DEEP-ONLY"
        ) and complete_metrics(args.comparison_root / "INLA-ONLY" / "INLA-TS") and complete_metrics(
            args.comparison_root / "DEEP-INLA" / "combined"
        ):
            run(
                [
                    python_bin,
                    CODE / "scripts/make_comparison_h3_outbreak_figure.py",
                    "--results-root",
                    args.comparison_root,
                    "--output",
                    ROOT / "paper/figures/Fig_H3_predicted_outbreak_examples",
                ],
                env=env,
                cwd=CODE,
            )
        elif not args.skip_figures:
            log("SKIP Fig_H3 because one or more core models are incomplete")

        write_status(status_path, f"COMPLETE finished={stamp()}")
        log("COMPLETE refactor model suite")
    except Exception as exc:
        write_status(status_path, f"FAILED finished={stamp()} error={exc}")
        log(f"FAILED {exc}")
        raise


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
