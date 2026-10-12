#!/usr/bin/env python3
"""Summarize commune and RHC alerts against native-scale MOH ground truth."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return f"external:{resolved.name}"


def metrics(labels, predictions):
    labels = np.asarray(labels, bool); predictions = np.asarray(predictions, bool)
    tp = int(np.sum(labels & predictions)); fp = int(np.sum(~labels & predictions))
    fn = int(np.sum(labels & ~predictions)); tn = int(np.sum(~labels & ~predictions))
    safe = lambda a, b: float(a / b) if b else 0.0
    precision = safe(tp, tp + fp); sensitivity = safe(tp, tp + fn)
    specificity = safe(tn, tn + fp)
    return {
        "n": len(labels), "outbreak_prevalence": float(labels.mean()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": safe(tp + tn, len(labels)), "precision": precision,
        "sensitivity": sensitivity, "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "f1": safe(2 * precision * sensitivity, precision + sensitivity),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commune", type=Path, required=True)
    parser.add_argument("--rhc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    commune = pd.read_csv(args.commune)
    rhc = pd.read_csv(args.rhc)
    rows = []
    c = commune.loc[commune.groundtruth.notna()]
    rows.append({
        "spatial_scale": "commune", "method": "Commune hybrid distribution alert",
        "groundtruth": "commune MOH outbreak threshold (+2 SD)", **metrics(c.groundtruth, c.commune_alert),
    })
    r = rhc.loc[rhc.moh_groundtruth.notna()]
    for method, column in [
        ("Original DeepAR-NB re-scored with MOH", "original_moh_outbreak_alert"),
        ("Proposed hybrid re-scored with MOH", "proposed_moh_outbreak_alert"),
        ("Original stored outbreak probability", "original_stored_outbreak_alert"),
        ("Current intervention flag", "current_intervention_flag"),
    ]:
        rows.append({
            "spatial_scale": "RHC", "method": method,
            "groundtruth": "RHC MOH outbreak threshold (+2 SD)", **metrics(r.moh_groundtruth, r[column]),
        })
    output = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "native_scale_alert_metrics.csv"
    output.to_csv(metrics_path, index=False)
    notes = {
        "primary_head_to_head": (
            "Compare original DeepAR-NB re-scored with MOH and proposed hybrid re-scored "
            "with MOH on the same 640 "
            "RHC-month rows and the same harmonized RHC MOH ground truth."
        ),
        "commune_caveat": (
            "Commune metrics use commune-specific MOH ground truth and are native-scale diagnostics; "
            "they are not directly rank-comparable with RHC metrics because aggregation changes event prevalence."
        ),
        "current_alert_caveat": (
            "The stored outbreak probability and intervention flag were developed under another outbreak definition. Their scores here are a "
            "retrospective harmonization to MOH ground truth, not a validation under its native definition."
        ),
        "evaluation_months": ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08"],
        "forecast_only_months": ["2026-09", "2026-10", "2026-11"],
    }
    notes_path = args.output_dir / "comparison_notes.json"
    notes_path.write_text(
        json.dumps(notes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    checksum_paths = [args.commune, args.rhc, metrics_path, notes_path]
    checksums = {portable_path(path): sha256(path) for path in checksum_paths}
    (args.output_dir / "CHECKSUMS_SHA256.json").write_text(
        json.dumps(checksums, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
