#!/usr/bin/env python3
"""Export auditable commune probabilities and fair alert comparisons.

The script deliberately keeps the source artifact identity in every row.  This
prevents a recovered archive from being silently presented as the locked paper
run when the corresponding raw files are no longer available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from analyze_province_scale import (
    apply_platt,
    calibration_diagnostic,
    classification,
    fit_platt,
    threshold_for_constraint,
)


KEYS = ["fcode", "issue_date", "target_date", "horizon"]
MODEL_ORDER = ["INLA-TS", "TCN-GNN", "DeepST-Alert", "DeepST-INLA"]

mpl.rcParams.update({
    "font.size": 12.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 10.5,
})
KEEP = KEYS + [
    "target_cases", "population", "outbreak_threshold", "outbreak_probability",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, model: str, split: str, source_status: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"fcode": str})
    missing = set(KEEP) - set(frame)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    output = frame[KEEP].copy()
    output["model"] = model
    output["split"] = split
    output["source_status"] = source_status
    output["source_sha256"] = sha256(path)
    output["observed_outbreak"] = (
        output.target_cases.to_numpy(float) > output.outbreak_threshold.to_numpy(float)
    ).astype(int)
    return output.sort_values(KEYS).reset_index(drop=True)


def analyse(validation: pd.DataFrame, test: pd.DataFrame):
    policies, matched, calibration = [], [], []
    calibrated_validation, calibrated_test = [], []
    for model in sorted(test.model.unique()):
        for horizon in (1, 2, 3):
            valid = validation[(validation.model == model) & validation.horizon.eq(horizon)].copy()
            target = test[(test.model == model) & test.horizon.eq(horizon)].copy()
            if len(valid) == 0 or len(target) == 0:
                raise ValueError(f"Missing {model} H{horizon}")
            intercept, slope = fit_platt(
                valid.outbreak_probability.to_numpy(float),
                valid.observed_outbreak.to_numpy(float),
            )
            valid["raw_outbreak_probability"] = valid.outbreak_probability
            target["raw_outbreak_probability"] = target.outbreak_probability
            valid["outbreak_probability"] = apply_platt(
                valid.raw_outbreak_probability, intercept, slope
            )
            target["outbreak_probability"] = apply_platt(
                target.raw_outbreak_probability, intercept, slope
            )
            policies.append({
                "model": model, "horizon": horizon,
                "validation_platt_intercept": intercept,
                "validation_platt_slope": slope,
                "selection_split": "2019-01 to 2020-12",
            })
            ty = target.observed_outbreak.to_numpy(bool)
            tp = target.outbreak_probability.to_numpy(float)
            vy = valid.observed_outbreak.to_numpy(bool)
            vp = valid.outbreak_probability.to_numpy(float)
            row = {
                "model": model, "horizon": horizon, "n_test": len(target),
                "test_prevalence": float(ty.mean()),
                "test_auprc": float(average_precision_score(ty, tp)),
                "test_brier": float(np.mean((tp - ty) ** 2)),
            }
            for specificity in (0.90, 0.95):
                cutoff, achieved = threshold_for_constraint(
                    vy, vp, specificity=specificity
                )
                scored = classification(ty, tp, cutoff)
                suffix = str(int(100 * specificity))
                row[f"validation_cutoff_at_specificity_{suffix}"] = cutoff
                row[f"validation_specificity_at_cutoff_{suffix}"] = achieved[
                    "outbreak_specificity"
                ]
                row[f"test_sensitivity_at_specificity_{suffix}"] = scored[
                    "outbreak_sensitivity"
                ]
                row[f"test_specificity_at_specificity_{suffix}"] = scored[
                    "outbreak_specificity"
                ]
            cutoff, achieved = threshold_for_constraint(vy, vp, sensitivity=0.80)
            scored = classification(ty, tp, cutoff)
            row.update({
                "validation_cutoff_at_sensitivity_80": cutoff,
                "validation_sensitivity_at_cutoff_80": achieved["outbreak_sensitivity"],
                "test_specificity_at_sensitivity_80": scored["outbreak_specificity"],
                "test_sensitivity_at_sensitivity_80": scored["outbreak_sensitivity"],
            })
            matched.append(row)

            for label, probability in (
                ("raw", target.raw_outbreak_probability.to_numpy(float)),
                ("common_validation_platt", tp),
            ):
                diag_intercept, diag_slope, ece = calibration_diagnostic(ty, probability)
                calibration.append({
                    "model": model, "horizon": horizon, "probability_version": label,
                    "n": len(target), "prevalence": float(ty.mean()),
                    "brier": float(np.mean((probability - ty) ** 2)),
                    "test_calibration_intercept": diag_intercept,
                    "test_calibration_slope": diag_slope,
                    "ece_5_equal_count_bins": ece,
                })
            calibrated_validation.append(valid)
            calibrated_test.append(target)
    return (
        pd.concat(calibrated_validation, ignore_index=True),
        pd.concat(calibrated_test, ignore_index=True),
        pd.DataFrame(policies), pd.DataFrame(matched), pd.DataFrame(calibration),
    )


def parse_model(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected MODEL=VALIDATION_CSV=TEST_CSV")
    return parts[0], Path(parts[1]), Path(parts[2])


def plot_comparison(test: pd.DataFrame, matched: pd.DataFrame, output: Path) -> None:
    """Supplementary Fig. S3 in the shared Nature Communications figure system
    (180 mm wide, 5-7 pt Arial, validated model colours, no in-figure title)."""
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6.5, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False,
    })
    ink, track = "#1F2430", "#ECEEF1"
    colors = dict(zip(MODEL_ORDER, ["#009E73", "#3F74AD", "#D55E00", "#7B2D8E"]))
    markers = dict(zip(MODEL_ORDER, ["s", "^", "D", "o"]))
    labels = {"INLA-TS": "Structured Bayesian", "TCN-GNN": "Neural count",
              "DeepST-Alert": "Neural warning", "DeepST-INLA": "DeepST\u2013INLA"}
    mm = 1 / 25.4
    fig = plt.figure(figsize=(180 * mm, 105 * mm))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.62, wspace=0.28,
                            left=0.075, right=0.985, bottom=0.14, top=0.93)
    for index, horizon in enumerate((1, 2, 3)):
        ax = fig.add_subplot(grid[0, index])
        ax.plot([0, 0.8], [0, 0.8], color="#8A919C", lw=0.7, ls=(0, (3, 2)), zorder=1)
        for model in MODEL_ORDER:
            part = test[(test.model.eq(model)) & (test.horizon.eq(horizon))]
            probability = part.outbreak_probability.to_numpy(float)
            observed = part.observed_outbreak.to_numpy(float)
            bins = np.array_split(np.argsort(probability), 8)
            ax.plot([probability[r].mean() for r in bins], [observed[r].mean() for r in bins],
                    marker=markers[model], ms=3, lw=1.0, color=colors[model],
                    mec="white", mew=0.4, label=labels[model], zorder=3)
        ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.8)
        ax.set_xlabel("Predicted probability")
        if index == 0:
            ax.set_ylabel("Observed frequency")
        ax.grid(color=track, lw=0.6); ax.set_axisbelow(True)
        ax.annotate("abc"[index], xy=(0, 1), xycoords="axes fraction", xytext=(-22 if index == 0 else -16, 6),
                    textcoords="offset points", fontsize=8, fontweight="bold", annotation_clip=False)
        ax.annotate(f"H{horizon}", xy=(0, 1), xycoords="axes fraction",
                    xytext=(-13 if index == 0 else -7, 6.6), textcoords="offset points",
                    fontsize=6.8, color=ink, annotation_clip=False)
    ax = fig.add_subplot(grid[1, :2])
    x = np.arange(len(MODEL_ORDER)); width = 0.26
    mean = matched.groupby("model", as_index=True).mean(numeric_only=True).reindex(MODEL_ORDER)
    shades = ("#5C6470", "#9AA1AC", "#CDD1D8")
    for offset, column, label, shade in (
        (-width, "test_auprc", "AUPRC", shades[0]),
        (0, "test_sensitivity_at_specificity_90", "Sensitivity at specificity \u2265 0.90", shades[1]),
        (width, "test_sensitivity_at_specificity_95", "Sensitivity at specificity \u2265 0.95", shades[2]),
    ):
        ax.bar(x + offset, mean[column], width=width * 0.92, color=shade, label=label)
    ax.set_xticks(x, [labels[m] for m in MODEL_ORDER])
    ax.set_ylim(0, 0.75); ax.set_ylabel("Mean H1\u2013H3 test value")
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.16),
              handlelength=1, columnspacing=1.2)
    ax.grid(axis="y", color=track, lw=0.6); ax.set_axisbelow(True)
    ax.annotate("d", xy=(0, 1.16), xycoords="axes fraction", xytext=(-22, 6),
                textcoords="offset points", fontsize=8, fontweight="bold", annotation_clip=False)
    ax = fig.add_subplot(grid[1, 2])
    ax.axis("off")
    handles, names = fig.axes[0].get_legend_handles_labels()
    ax.legend(handles, names, frameon=False, loc="center left", title="Model system",
              title_fontsize=6.5, handlelength=1.8)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, type=parse_model)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-status", default="recovered_archive_not_locked_current",
        help="Provenance label written into every exported row.",
    )
    args = parser.parse_args()
    validation, test, sources = [], [], []
    for model, validation_path, test_path in args.model:
        validation.append(load(validation_path, model, "validation", args.source_status))
        test.append(load(test_path, model, "test", args.source_status))
        sources.append({
            "model": model,
            "validation_path": str(validation_path.resolve()),
            "validation_sha256": sha256(validation_path),
            "test_path": str(test_path.resolve()),
            "test_sha256": sha256(test_path),
        })
    validation_frame = pd.concat(validation, ignore_index=True)
    test_frame = pd.concat(test, ignore_index=True)
    reference = test_frame[test_frame.model.eq(test_frame.model.iloc[0])][KEYS].reset_index(drop=True)
    reference_targets = test_frame[test_frame.model.eq(test_frame.model.iloc[0])][
        ["target_cases", "population", "outbreak_threshold"]
    ].reset_index(drop=True)
    for model, part in test_frame.groupby("model"):
        if not part[KEYS].reset_index(drop=True).equals(reference):
            raise ValueError(f"Test keys do not align for {model}")
        values = part[["target_cases", "population", "outbreak_threshold"]].reset_index(drop=True)
        if not np.allclose(values, reference_targets, rtol=0, atol=1e-6):
            raise ValueError(f"Test outcomes/thresholds do not align for {model}")
    calibrated_valid, calibrated_test, policies, matched, calibration = analyse(
        validation_frame, test_frame
    )
    args.output.mkdir(parents=True, exist_ok=True)
    validation_frame.to_csv(args.output / "row_probabilities_validation_raw.csv", index=False)
    test_frame.to_csv(args.output / "row_probabilities_test_raw.csv", index=False)
    calibrated_valid.to_csv(
        args.output / "row_probabilities_validation_common_calibration.csv", index=False
    )
    calibrated_test.to_csv(
        args.output / "row_probabilities_test_common_calibration.csv", index=False
    )
    policies.to_csv(args.output / "validation_calibration_policies.csv", index=False)
    matched.to_csv(args.output / "matched_operating_points.csv", index=False)
    calibration.to_csv(args.output / "calibration_comparison.csv", index=False)
    summary = matched.groupby("model", as_index=False).mean(numeric_only=True)
    calibrated_summary = calibration[
        calibration.probability_version.eq("common_validation_platt")
    ].groupby("model", as_index=False).mean(numeric_only=True)
    summary = summary.merge(
        calibrated_summary[[
            "model", "brier", "test_calibration_intercept", "test_calibration_slope",
            "ece_5_equal_count_bins",
        ]], on="model", how="left",
    )
    summary.to_csv(args.output / "matched_calibration_mean_H1_H3.csv", index=False)
    plot_comparison(
        calibrated_test, matched, args.output / "commune_reliability_matched.pdf"
    )
    manifest = {
        "source_status": args.source_status,
        "warning": (
            "These recovered fixed-split artifacts are internally coherent but are not "
            "the missing raw artifacts behind results/current/final_metrics_by_horizon.csv."
        ),
        "models": sources,
        "validation_rows": len(validation_frame), "test_rows": len(test_frame),
        "key_columns": KEYS,
        "analysis": "validation Platt calibration; validation-selected matched constraints",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}: validation={len(validation_frame):,}, test={len(test_frame):,}")


if __name__ == "__main__":
    main()
