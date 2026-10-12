#!/usr/bin/env python3
"""Phase D: compare six locked systems under one validation-only alert protocol."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from analyze_province_scale import apply_platt, calibration_diagnostic, classification, fit_platt


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/recovered_from_trash/results/model_runs_train2010_2018_valid2019_2020_test2021_2025_external_thresholds"
OUT = ROOT / "results/rev_20261012/matched_operating_point"
SEED = 20260917
REPLICATES = 2000
KEYS = ["fcode", "issue_date", "target_date", "horizon"]
MODELS = {
    "INLA-TS": (
        ART / "INLA-ONLY-validation/INLA-TS/predictions.csv",
        ART / "INLA-ONLY-test/INLA-TS/predictions.csv",
        "outbreak_probability",
    ),
    "Representation-INLA": (
        ART / "REPRESENTATION-INLA/validation/combined/predictions.csv",
        ART / "REPRESENTATION-INLA/test/combined/predictions.csv",
        "outbreak_probability",
    ),
    "TCN-GNN Base": (
        ART / "TCN-GNN-BASE/validation_predictions.csv",
        ART / "TCN-GNN-BASE/predictions.csv",
        "outbreak_probability_samples",
    ),
    "DeepST-Alert": (
        ART / "DEEPST-ALERT/validation_predictions.csv",
        ART / "DEEPST-ALERT/predictions.csv",
        "outbreak_probability_auxiliary",
    ),
    "DeepST-INLA": (
        ART / "DEEPST-INLA/validation_predictions.csv",
        ART / "DEEPST-INLA/predictions.csv",
        "outbreak_probability",
    ),
    "LSTM-GNN": (
        ART / "LSTM-GNN/validation_predictions.csv",
        ART / "LSTM-GNN/predictions.csv",
        "outbreak_probability_samples",
    ),
}
METRICS = ["brier", "sensitivity", "specificity", "precision", "f1", "auprc"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path, probability_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"fcode": str})
    required = set(KEYS + ["target_cases", "outbreak_threshold", probability_column])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("%s missing %s" % (path, sorted(missing)))
    out = frame[KEYS + ["target_cases", "outbreak_threshold", probability_column]].copy()
    out = out.rename(columns={probability_column: "raw_probability"})
    out["issue_date"] = pd.to_datetime(out.issue_date)
    out["target_date"] = pd.to_datetime(out.target_date)
    out["label"] = out.target_cases.to_numpy(float) > out.outbreak_threshold.to_numpy(float)
    return out.sort_values(KEYS).reset_index(drop=True)


def cutoff_rows(labels: np.ndarray, probability: np.ndarray):
    """Return exact metrics at every observed cutoff in O(n log n)."""
    labels = labels.astype(bool)
    order = np.argsort(-probability, kind="mergesort")
    p, y = probability[order], labels[order]
    ends = np.r_[np.flatnonzero(p[:-1] != p[1:]), len(p) - 1]
    tp = np.cumsum(y)[ends].astype(float)
    fp = np.cumsum(~y)[ends].astype(float)
    positives, negatives = float(y.sum()), float((~y).sum())
    fn, tn = positives - tp, negatives - fp
    divide = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    precision = divide(tp, tp + fp)
    f1 = divide(2 * precision * sensitivity, precision + sensitivity)
    rows = pd.DataFrame({"cutoff": p[ends], "outbreak_sensitivity": sensitivity,
                         "outbreak_specificity": specificity, "outbreak_precision": precision,
                         "outbreak_f1": f1})
    # Include the two boundary candidates used by the original exhaustive code.
    for cutoff in (0.0, 1.0):
        if not np.any(rows.cutoff.eq(cutoff)):
            score = classification(labels, probability, cutoff)
            rows = pd.concat([rows, pd.DataFrame([{"cutoff": cutoff, **score}])], ignore_index=True)
    return rows


def choose_cutoff(labels: np.ndarray, probability: np.ndarray):
    brier = float(np.mean((probability - labels) ** 2))
    rows = cutoff_rows(labels, probability)
    eligible = rows[(rows.outbreak_precision >= 0.5) & (rows.outbreak_specificity >= 0.9)].copy()
    if not eligible.empty:
        values = np.clip(eligible[["outbreak_precision", "outbreak_specificity",
                                  "outbreak_sensitivity", "outbreak_f1"]].to_numpy(), 1e-12, 1)
        eligible["objective"] = (np.prod(values, axis=1) * max(1e-12, 1 - brier)) ** 0.2
        winner = eligible.sort_values(["objective", "outbreak_f1", "outbreak_sensitivity", "cutoff"]).iloc[-1]
        return float(winner.cutoff), "multiobjective", winner.to_dict()
    winner = rows.sort_values(["outbreak_f1", "outbreak_sensitivity", "cutoff"]).iloc[-1]
    return float(winner.cutoff), "fallback_maximum_f1", winner.to_dict()


def constraint_cutoff(labels: np.ndarray, probability: np.ndarray, target: float):
    eligible = cutoff_rows(labels, probability)
    eligible = eligible[eligible.outbreak_specificity >= target]
    winner = eligible.sort_values(["outbreak_sensitivity", "cutoff"]).iloc[-1]
    return float(winner.cutoff), winner.to_dict()


def ece(labels: np.ndarray, probability: np.ndarray, bins: int) -> float:
    chunks = np.array_split(np.argsort(probability), bins)
    return float(sum(len(idx) / len(labels) * abs(labels[idx].mean() - probability[idx].mean())
                     for idx in chunks if len(idx)))


def values(frame: pd.DataFrame, weights: np.ndarray | None = None):
    y = frame.label.to_numpy(bool)
    p = frame.probability.to_numpy(float)
    alert = p >= frame.cutoff.to_numpy(float)
    w = np.ones(len(frame), float) if weights is None else weights.astype(float)
    tp = np.sum(w * (y & alert)); fp = np.sum(w * (~y & alert))
    fn = np.sum(w * (y & ~alert)); tn = np.sum(w * (~y & ~alert))
    ratio = lambda a, b: float(a / b) if b else 0.0
    sens, spec, prec = ratio(tp, tp + fn), ratio(tn, tn + fp), ratio(tp, tp + fp)
    return {
        "brier": float(np.sum(w * (p - y) ** 2) / np.sum(w)),
        "sensitivity": sens, "specificity": spec, "precision": prec,
        "f1": ratio(2 * prec * sens, prec + sens),
        "auprc": float(average_precision_score(y, p, sample_weight=w)),
    }


def calibrate():
    policies, frames, point_rows = [], {}, []
    reference = None
    for model, (valid_path, test_path, probability_column) in MODELS.items():
        valid, test = read(valid_path, probability_column), read(test_path, probability_column)
        if reference is None:
            reference = test[KEYS + ["target_cases", "outbreak_threshold", "label"]]
        else:
            if not test[KEYS].equals(reference[KEYS]):
                raise ValueError("Test keys differ for %s" % model)
            if not np.allclose(test[["target_cases", "outbreak_threshold"]],
                               reference[["target_cases", "outbreak_threshold"]], rtol=0, atol=1e-7):
                raise ValueError("Outcomes/thresholds differ for %s" % model)
        pieces = []
        for horizon in (1, 2, 3):
            v = valid[valid.horizon.eq(horizon)].copy()
            t = test[test.horizon.eq(horizon)].copy()
            intercept, slope = fit_platt(v.raw_probability.to_numpy(float), v.label.to_numpy(float))
            vp = apply_platt(v.raw_probability.to_numpy(float), intercept, slope)
            tp = apply_platt(t.raw_probability.to_numpy(float), intercept, slope)
            cutoff, rule, achieved = choose_cutoff(v.label.to_numpy(bool), vp)
            c90, a90 = constraint_cutoff(v.label.to_numpy(bool), vp, 0.90)
            c95, a95 = constraint_cutoff(v.label.to_numpy(bool), vp, 0.95)
            t["probability"] = tp; t["cutoff"] = cutoff; t["model"] = model
            pieces.append(t)
            diag_i, diag_s, _ = calibration_diagnostic(t.label.to_numpy(float), tp)
            row = {"model": model, "scope": "H%d" % horizon, "horizon": horizon,
                   "n": len(t), **values(t), "calibration_intercept": diag_i,
                   "calibration_slope": diag_s, "ece_5": ece(t.label.to_numpy(bool), tp, 5),
                   "ece_10": ece(t.label.to_numpy(bool), tp, 10), "cutoff": cutoff,
                   "cutoff_rule": rule}
            point_rows.append(row)
            policies.append({
                "model": model, "horizon": horizon, "probability_source": probability_column,
                "platt_intercept": intercept, "platt_slope": slope, "cutoff": cutoff,
                "cutoff_rule": rule, "validation_precision": achieved["outbreak_precision"],
                "validation_specificity": achieved["outbreak_specificity"],
                "validation_sensitivity": achieved["outbreak_sensitivity"],
                "validation_f1": achieved["outbreak_f1"],
                "specificity_90_cutoff": c90, "specificity_90_achieved": a90["outbreak_specificity"],
                "specificity_95_cutoff": c95, "specificity_95_achieved": a95["outbreak_specificity"],
            })
        frames[model] = pd.concat(pieces, ignore_index=True)
    return pd.DataFrame(policies), frames, pd.DataFrame(point_rows)


def bootstrap_score(frame, counts, row_month):
    """Vectorised block bootstrap, including an exact weighted ranking AP."""
    y = frame.label.to_numpy(bool)
    p = frame.probability.to_numpy(float)
    alert = p >= frame.cutoff.to_numpy(float)
    month_n = np.bincount(row_month, minlength=counts.shape[1])
    aggregates = {}
    for name, contribution in {
        "tp": y & alert, "fp": (~y) & alert, "fn": y & (~alert), "tn": (~y) & (~alert),
        "brier_sum": (p - y) ** 2,
    }.items():
        aggregates[name] = np.bincount(row_month, weights=np.asarray(contribution, float),
                                       minlength=counts.shape[1])
    tp, fp = counts @ aggregates["tp"], counts @ aggregates["fp"]
    fn, tn = counts @ aggregates["fn"], counts @ aggregates["tn"]
    divide = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    sensitivity, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
    precision = divide(tp, tp + fp)
    output = {
        "brier": divide(counts @ aggregates["brier_sum"], counts @ month_n),
        "sensitivity": sensitivity, "specificity": specificity, "precision": precision,
        "f1": divide(2 * precision * sensitivity, precision + sensitivity),
    }
    order = np.argsort(-p, kind="mergesort")
    ys, ms = y[order], row_month[order]
    group_ends = np.r_[np.flatnonzero(p[order][:-1] != p[order][1:]), len(order) - 1]
    ap = np.empty(len(counts))
    for start in range(0, len(counts), 50):
        block = counts[start:start + 50, ms]
        positive = block * ys
        cumulative_positive = np.cumsum(positive, axis=1)
        cumulative_total = np.cumsum(block, axis=1)
        cumulative_positive = cumulative_positive[:, group_ends]
        cumulative_total = cumulative_total[:, group_ends]
        group_positive = np.diff(cumulative_positive, axis=1,
                                 prepend=np.zeros((len(block), 1)))
        precision_at_group = np.divide(cumulative_positive, cumulative_total,
                                       out=np.zeros_like(cumulative_positive, float),
                                       where=cumulative_total != 0)
        total_positive = positive.sum(axis=1)
        ap[start:start + len(block)] = np.divide(
            (precision_at_group * group_positive).sum(axis=1), total_positive,
            out=np.zeros(len(block)), where=total_positive != 0)
    output["auprc"] = ap
    return output


def bootstrap(frames, point):
    rng = np.random.default_rng(SEED)
    draws = {model: {metric: np.zeros(REPLICATES) for metric in METRICS} for model in MODELS}
    for horizon in (1, 2, 3):
        reference = frames["DeepST-INLA"].query("horizon == @horizon")
        months = reference.target_date.dt.to_period("M").astype(str).to_numpy()
        unique = np.unique(months)
        month_index = {month: index for index, month in enumerate(unique)}
        row_month = np.array([month_index[value] for value in months])
        counts = np.stack([np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
                           for _ in range(REPLICATES)])
        for model, frame in frames.items():
            part = frame.query("horizon == @horizon")
            score = bootstrap_score(part, counts, row_month)
            for metric in METRICS:
                draws[model][metric] += score[metric] / 3.0
    mean_rows, contrast_rows = [], []
    for model, frame in frames.items():
        model_points = point[point.model.eq(model)]
        row = {"model": model, "scope": "mean_H1_H3", "horizon": np.nan, "n": len(frame)}
        for metric in METRICS:
            estimate = float(model_points[metric].mean())
            sample = draws[model][metric]
            low, high = np.quantile(sample, [0.025, 0.975])
            row[metric] = estimate; row[metric + "_sd"] = float(sample.std(ddof=1))
            row[metric + "_ci025"] = low; row[metric + "_ci975"] = high
        for metric in ("calibration_intercept", "calibration_slope", "ece_5", "ece_10"):
            row[metric] = float(model_points[metric].mean())
        mean_rows.append(row)
        if model != "DeepST-INLA":
            for metric in METRICS:
                estimate = float(point[point.model.eq(model)][metric].mean() -
                                 point[point.model.eq("DeepST-INLA")][metric].mean())
                sample = draws[model][metric] - draws["DeepST-INLA"][metric]
                low, high = np.quantile(sample, [0.025, 0.975])
                contrast_rows.append({
                    "comparator": model, "reference": "DeepST-INLA", "metric": metric,
                    "difference_comparator_minus_reference": estimate,
                    "bootstrap_sd": float(sample.std(ddof=1)), "ci025": low, "ci975": high,
                    "replicates": REPLICATES, "seed": SEED,
                })
    return pd.DataFrame(mean_rows), pd.DataFrame(contrast_rows)


def capacity(frames):
    budgets = (0.05, 0.10, 0.20, 0.30)
    rng = np.random.default_rng(SEED)
    rows, draw_store = [], {}
    for horizon in (1, 2, 3):
        reference = frames["DeepST-INLA"].query("horizon == @horizon")
        months = sorted(reference.target_date.dt.to_period("M").unique())
        bootstrap_indices = np.stack([rng.integers(0, len(months), len(months))
                                      for _ in range(REPLICATES)])
        for model, frame in frames.items():
            part = frame[frame.horizon.eq(horizon)]
            for budget in budgets:
                captured, total = [], []
                for month in months:
                    block = part[part.target_date.dt.to_period("M").eq(month)]
                    k = max(1, int(np.ceil(len(block) * budget)))
                    selected = block.nlargest(k, "probability")
                    captured.append(int(selected.label.sum())); total.append(int(block.label.sum()))
                captured, total = np.asarray(captured), np.asarray(total)
                estimate = float(captured.sum() / total.sum())
                numerator = captured[bootstrap_indices].sum(axis=1)
                denominator = total[bootstrap_indices].sum(axis=1)
                boot = np.divide(numerator, denominator, out=np.zeros(REPLICATES), where=denominator != 0)
                draw_store[(model, horizon, budget)] = boot
                low, high = np.quantile(boot, [0.025, 0.975])
                rows.append({"model": model, "horizon": horizon, "budget": budget,
                             "capture": estimate, "bootstrap_sd": boot.std(ddof=1),
                             "ci025": low, "ci975": high, "replicates": REPLICATES, "seed": SEED})
    output = pd.DataFrame(rows)
    ref = output[output.model.eq("DeepST-INLA")].set_index(["horizon", "budget"])
    output["difference_from_deepst_inla"] = [
        row.capture - ref.loc[(row.horizon, row.budget), "capture"] for row in output.itertuples()
    ]
    output["difference_ci025"] = [
        np.quantile(draw_store[(row.model, row.horizon, row.budget)] -
                    draw_store[("DeepST-INLA", row.horizon, row.budget)], 0.025)
        for row in output.itertuples()
    ]
    output["difference_ci975"] = [
        np.quantile(draw_store[(row.model, row.horizon, row.budget)] -
                    draw_store[("DeepST-INLA", row.horizon, row.budget)], 0.975)
        for row in output.itertuples()
    ]
    return output


def write_csv(frame: pd.DataFrame, path: Path, inputs):
    frame.to_csv(path, index=False)
    provenance = {
        "inputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in inputs],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "command": " ".join(sys.argv), "seeds": [SEED],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selection_outcomes": "2019-2020 only", "test_outcomes_used_for_selection": False,
    }
    path.with_suffix(path.suffix + ".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def write_table(mean: pd.DataFrame):
    order = list(MODELS)
    mean = mean.set_index("model").loc[order]
    cols = [("brier", False), ("sensitivity", True), ("specificity", True),
            ("precision", True), ("f1", True), ("auprc", True)]
    best = {name: (mean[name].max() if high else mean[name].min()) for name, high in cols}
    lines = ["\\resizebox{\\textwidth}{!}{\\begin{tabular}{lcccccc}", "\\toprule",
             "Model & Brier & Sensitivity & Specificity & Precision & F1 & AUPRC\\\\", "\\midrule"]
    labels = {"Representation-INLA": "Representation--INLA", "TCN-GNN Base": "TCN--GNN Base",
              "DeepST-Alert": "DeepST--Alert", "DeepST-INLA": "DeepST--INLA",
              "LSTM-GNN": "LSTM--GNN", "INLA-TS": "INLA--TS"}
    for model, row in mean.iterrows():
        cells = []
        for name, _ in cols:
            value = row[name]
            text = "%.3f\\,$\\pm$\\,%.3f" % (value, row[name + "_sd"])
            if np.isclose(value, best[name]):
                text = "\\textbf{%s}" % text
            cells.append(text)
        lines.append(labels[model] + " & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}}"]
    (ROOT / "paper/generated/table_matched_protocol.tex").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    policies, frames, point = calibrate()
    mean, contrasts = bootstrap(frames, point)
    capture = capacity(frames)
    inputs = [item for values in MODELS.values() for item in values[:2]]
    write_csv(point, OUT / "metrics_by_horizon.csv", inputs)
    write_csv(mean, OUT / "metrics_mean.csv", inputs)
    write_csv(contrasts, OUT / "paired_contrasts.csv", inputs)
    write_csv(capture, OUT / "capture.csv", inputs)
    write_csv(policies, OUT / "validation_policies.csv", inputs)
    write_table(mean)
    print(mean[["model", "brier", "sensitivity", "specificity", "precision", "f1", "auprc"]].to_string(index=False))


if __name__ == "__main__":
    main()
