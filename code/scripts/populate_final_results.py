#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
FINAL_DIR = RESULTS_DIR / "final"

COMPLETE_MODELS = [
    "seasonal_naive",
    "province_allocation",
    "INLA-T",
    "INLA-S",
    "INLA-TS",
    "INLA-T-AR1",
    "INLA-ST-AR1",
    "CNN",
    "LSTM",
    "CNN-LSTM",
    "GNN",
    "LSTM-FULL",
    "LSTM-GNN",
    "CNN-LSTM-FULL",
    "CNN-LSTM-GNN",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M6-INLA",
    "TCN-H64",
    "TCN-L24",
]

PENDING_MODELS = [
    "INLA-ST-DYN",
]

REQUIRED_FILES = ("metrics.csv", "predictions.csv")


def model_ready(source: Path) -> bool:
    return all((source / name).is_file() for name in REQUIRED_FILES)


def copy_model(model: str) -> dict:
    source = RESULTS_DIR / model
    target = FINAL_DIR / model
    if not model_ready(source):
        return {"model": model, "status": "skipped", "reason": "missing metrics or predictions"}
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    return {"model": model, "status": "copied", "source": str(source)}


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    skipped = []
    for model in COMPLETE_MODELS:
        outcome = copy_model(model)
        (copied if outcome["status"] == "copied" else skipped).append(outcome)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Locked paper-ready commune benchmark results",
        "evaluation_split": {
            "train": "2010-01-01 to 2020-12-01",
            "validation": "2021-01-01 to 2022-12-01",
            "test": "2023-01-01 to 2025-12-01",
        },
        "copied_models": [item["model"] for item in copied],
        "skipped_models": skipped,
        "pending_models": PENDING_MODELS,
        "missing_v1_inla_baselines": [
            "inla",
            "inla_temporal",
            "inla_spatial",
            "inla_spatiotemporal",
        ],
    }
    (FINAL_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
