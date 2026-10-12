#!/usr/bin/env python3
"""Phase A: compare the historical five-year rule with audited MOH labels."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from analyze_spatial_support import build_primary_thresholds

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from dengue_h3.data import load_outbreak_thresholds  # noqa: E402

MONTHLY = ROOT / "code/data/processed/feature_suite/monthly_commune_open.csv"
AUDITED = ROOT / "code/data/raw/operational_2026/commune_monthly_thresholds.csv"
OUT = ROOT / "results/rev_20261012/label_audit"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    monthly = pd.read_csv(MONTHLY, dtype={"fcode": str},
                          usecols=["fcode", "year", "month", "obs_dengue_cases"])
    monthly["date"] = pd.to_datetime(dict(year=monthly.year, month=monthly.month, day=1))
    targets = pd.date_range("2021-01-01", "2025-12-01", freq="MS")
    old = build_primary_thresholds(monthly, targets).rename(
        columns={"moh_threshold_2sd_cases": "old_threshold"})
    audited = load_outbreak_thresholds(AUDITED).rename(
        columns={"outbreak_threshold": "audited_threshold"})
    outcome = monthly.rename(columns={"obs_dengue_cases": "target_cases"})[
        ["fcode", "date", "target_cases"]]
    joined = old.merge(audited, on=["fcode", "date"], how="outer", validate="one_to_one")
    joined = joined.merge(outcome, on=["fcode", "date"], how="left", validate="one_to_one")
    joined = joined[joined.date.between(targets.min(), targets.max())].copy()
    joined["old_label"] = joined.target_cases > joined.old_threshold
    joined["audited_label"] = joined.target_cases > joined.audited_threshold
    valid = joined.dropna(subset=["old_threshold", "audited_threshold", "target_cases"]).copy()
    summary = pd.DataFrame([{
        "scope": "unique commune-target-month keys, 2021-2025",
        "n_keys": len(valid),
        "old_prevalence": valid.old_label.mean(),
        "audited_prevalence": valid.audited_label.mean(),
        "agreement": (valid.old_label == valid.audited_label).mean(),
        "n_disagree": (valid.old_label != valid.audited_label).sum(),
        "old_positive_audited_negative": (valid.old_label & ~valid.audited_label).sum(),
        "old_negative_audited_positive": (~valid.old_label & valid.audited_label).sum(),
        "n_old_threshold_missing": joined.old_threshold.isna().sum(),
        "n_audited_threshold_missing": joined.audited_threshold.isna().sum(),
    }])
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {OUT / "label_agreement_summary.csv": summary,
               OUT / "label_disagreements.csv": valid[valid.old_label != valid.audited_label]}
    provenance = {
        "inputs": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)}
                   for p in (MONTHLY, AUDITED)],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "command": " ".join(sys.argv), "seeds": [],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "implementations": {
            "old": "code.paper_pipeline.analyze_spatial_support.build_primary_thresholds",
            "audited_loader": "dengue_h3.data.load_outbreak_thresholds",
        },
    }
    for path, frame in outputs.items():
        frame.to_csv(path, index=False)
        path.with_suffix(path.suffix + ".provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
