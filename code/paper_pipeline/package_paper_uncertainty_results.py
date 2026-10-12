#!/usr/bin/env python3
"""Arrange compact uncertainty results and create one paper-results archive."""
from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[2]
STAGING = ROOT / "results/figure7_multiseed_validation"
RESULTS = ROOT / "results/paper_uncertainty_20261004"
ABLATION = RESULTS / "ablation_multiseed_validation"
OUTPUT = ROOT / "dengue_paper_results_20261004.zip"


def copy(source: Path, destination: Path) -> None:
    if source.is_file() and source.stat().st_size:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    source = STAGING if STAGING.exists() else ABLATION
    if not (source / "COMPLETE").exists():
        raise RuntimeError("Five-seed ablation batch is not complete")
    RESULTS.mkdir(parents=True, exist_ok=True)
    if source == STAGING:
        if ABLATION.exists():
            shutil.rmtree(ABLATION)
        shutil.move(str(STAGING), str(ABLATION))

    (RESULTS / "README.md").write_text(
        "# Paper uncertainty results\n\n"
        f"Arranged {datetime.now().astimezone().isoformat()}.\n\n"
        "- `ablation_multiseed_validation/`: five seeds for every architecture "
        "and environmental-feature configuration. Models were trained on "
        "2010--2018 and evaluated on 2019--2020 validation data. The selected "
        "TCN-GNN/full-feature configuration is shared between the two panels, "
        "giving 10 unique configurations and 50 training runs.\n"
        "- `spatial_support_bootstrap/`: target-month block-bootstrap uncertainty "
        "for controlled spatial aggregation.\n"
        "- `main_model_uncertainty/`: main-model bootstrap intervals and final "
        "metric tables.\n"
        "- `previous_test_period_seed_summaries/`: earlier seed summaries retained "
        "for provenance and kept separate from the new validation ablations.\n"
        "- `reproducibility/`: configuration and scripts used for these results.\n\n"
        "Weights, predictive draws, row-level predictions, caches and restricted "
        "surveillance inputs are excluded.\n",
        encoding="utf-8",
    )

    spatial = ROOT / "results/spatial_support"
    for name in (
        "manifest.json", "metrics_by_scale_horizon.csv", "metrics_mean_by_scale.csv",
        "support_block_bootstrap.csv", "support_trend_tests.csv",
        "matched_operating_points.csv", "validation_calibration_policies.csv",
        "controlled_spatial_support.pdf",
    ):
        copy(spatial / name, RESULTS / "spatial_support_bootstrap" / name)

    complete = ROOT / "results/complete_uncertainty_2021_2025"
    for name in ("README.md", "NEURAL_COMPLETE", "HYBRID_COMPLETE", "INLA_READY"):
        copy(complete / name, RESULTS / "main_model_uncertainty" / name)
    for directory in ("bootstrap", "final"):
        for path in sorted((complete / directory).glob("*")):
            copy(path, RESULTS / "main_model_uncertainty" / directory / path.name)

    previous = ROOT / "results/uncertainty_all_experiments"
    for name in ("summary_all_scopes.csv", "summary_by_horizon.csv",
                 "summary_mean_H1_H3.csv", "table_mean_plus_minus_sd.csv", "status.json"):
        copy(previous / name, RESULTS / "previous_test_period_seed_summaries" / name)

    for relative in (
        "code/config/eval_2021_2025.yaml",
        "code/paper_pipeline/run_figure7_multiseed.py",
        "code/paper_pipeline/package_paper_uncertainty_results.py",
        "code/paper_pipeline/bootstrap_all_models.py",
        "code/scripts/run_figure7_multiseed_tmux.sh",
    ):
        copy(ROOT / relative, RESULTS / "reproducibility" / relative)

    for path in ROOT.glob("*.zip"):
        path.unlink()
    files = sorted(path for path in RESULTS.rglob("*") if path.is_file())
    manifest = []
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = Path(RESULTS.name) / path.relative_to(RESULTS)
            data = path.read_bytes()
            archive.writestr(str(relative), data)
            manifest.append(f"{hashlib.sha256(data).hexdigest()}  {relative}")
        archive.writestr(
            str(Path(RESULTS.name) / "CHECKSUMS_SHA256.txt"),
            "\n".join(manifest) + "\n",
        )
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    (RESULTS / "PACKAGE_COMPLETE").write_text(
        f"archive={OUTPUT.name}\nbytes={OUTPUT.stat().st_size}\nsha256={digest}\n",
        encoding="utf-8",
    )
    print(OUTPUT)
    print(f"files={len(files) + 1} bytes={OUTPUT.stat().st_size} sha256={digest}")


if __name__ == "__main__":
    main()
