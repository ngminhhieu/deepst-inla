#!/usr/bin/env python3
"""Build publication-grade commune-centroid spatial result maps from real outputs.

This script deliberately requires the private study centroids and real model
predictions. It never synthesizes locations or forecast values. Typical use:

python code/paper_pipeline/build_spatial_result_maps.py \
  --centroids code/data/processed/qc_v1/commune_centroids.csv \
  --prediction DeepST-INLA=results/final_multiobjective_model_2021_2025/predictions.csv \
  --prediction INLA-TS=results/eval_2021_2025_v1/INLA-ONLY-test/INLA-TS/predictions.csv \
  --target-date 2024-08-01 --horizon 3 \
  --decision-threshold DeepST-INLA=0.37 --decision-threshold INLA-TS=0.50 \
  --output paper/figures/Fig_spatial_case_study.pdf

Required prediction columns follow the repository contract:
fcode,target_date,horizon,target_cases,predicted_mean,outbreak_threshold,
outbreak_probability.  The generated panels show observed incidence, predicted
incidence, outbreak probability and alert/error category on the exact commune
centroids supplied by the study data.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

REQ = {"fcode", "target_date", "horizon", "target_cases", "predicted_mean", "outbreak_threshold", "outbreak_probability"}
OUTCOME_COLORS = ["#d9d9d9", "#e69f00", "#009e73", "#d55e00"]
OUTCOME_LABELS = ["True negative", "False positive", "True positive", "False negative"]

mpl.rcParams.update({
    "font.size": 14.5,
    "axes.titlesize": 13.5,
    "axes.labelsize": 14.5,
    "xtick.labelsize": 13.5,
    "ytick.labelsize": 13.5,
    "legend.fontsize": 11.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def kv(items):
    out = {}
    for item in items or []:
        key, value = item.split("=", 1)
        out[key] = value
    return out

def load_prediction(path: Path, target_date: str, horizon: int):
    f = pd.read_csv(path, dtype={"fcode": str})
    missing = REQ.difference(f.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    f["target_date"] = pd.to_datetime(f["target_date"]).dt.strftime("%Y-%m-%d")
    f = f[(f.target_date == target_date) & (f.horizon.astype(int) == horizon)].copy()
    if f.empty:
        raise ValueError(f"No rows for target_date={target_date}, horizon={horizon} in {path}")
    return f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centroids", type=Path, required=True)
    ap.add_argument("--prediction", action="append", required=True, help="MODEL=predictions.csv")
    ap.add_argument("--decision-threshold", action="append", default=[], help="MODEL=cutoff")
    ap.add_argument("--target-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--horizon", type=int, choices=[1,2,3], required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if not args.centroids.exists():
        raise FileNotFoundError(f"Exact study centroids are required: {args.centroids}")
    c = pd.read_csv(args.centroids, dtype={"fcode": str})
    for col in ("fcode", "longitude", "latitude"):
        if col not in c.columns:
            raise ValueError(f"Centroid file missing {col}")
    pred_paths = {k: Path(v) for k,v in kv(args.prediction).items()}
    cuts = {k: float(v) for k,v in kv(args.decision_threshold).items()}
    loaded = {name: load_prediction(path, args.target_date, args.horizon) for name,path in pred_paths.items()}

    # Observed panel uses the first model because all models share exact test keys.
    first = next(iter(loaded.values())).merge(c, on="fcode", how="inner")
    if len(first) < 10:
        raise ValueError("Too few predictions match the exact study centroids; check fcode encoding.")

    n_models = len(loaded)
    fig, axes = plt.subplots(n_models, 4, figsize=(13.2, 3.25*n_models), constrained_layout=True)
    if n_models == 1:
        axes = np.asarray([axes])
    for r,(name,p) in enumerate(loaded.items()):
        d = p.merge(c, on="fcode", how="inner")
        observed = d.target_cases.to_numpy(float)
        outbreak = observed > d.outbreak_threshold.to_numpy(float)
        cutoff = cuts.get(name, 0.5)
        alert = d.outbreak_probability.to_numpy(float) >= cutoff
        category = np.select([outbreak & alert, outbreak & ~alert, ~outbreak & alert], [2,3,1], default=0)
        outcome_cmap = ListedColormap(OUTCOME_COLORS)
        outcome_norm = BoundaryNorm(np.arange(-0.5, 4.5), outcome_cmap.N)
        panels = [
            (observed, "Observed cases", "viridis", None),
            (d.predicted_mean.to_numpy(float), f"{name}: predicted mean", "viridis", None),
            (d.outbreak_probability.to_numpy(float), f"{name}: outbreak probability", "magma", (0,1)),
            (category, f"{name}: alert outcomes", outcome_cmap, None),
        ]
        for j,(values,title,cmap,limits) in enumerate(panels):
            ax=axes[r,j]
            ax.text(-0.12, 1.08, chr(ord("a") + 4*r + j), transform=ax.transAxes,
                    fontsize=15.5, fontweight="bold", va="top", ha="left")
            kw={"c":values,"cmap":cmap,"s":20,"linewidths":0,"alpha":0.9}
            if limits: kw.update(vmin=limits[0],vmax=limits[1])
            if j == 3: kw["norm"] = outcome_norm
            sc=ax.scatter(d.longitude,d.latitude,**kw)
            ax.set_title(title, fontsize=13.5)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            ax.spines[["top","right"]].set_visible(False)
            if j < 3: fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
            else:
                handles = [
                    Line2D(
                        [0], [0], marker="o", linestyle="", markersize=5,
                        markerfacecolor=color, markeredgecolor="none", label=label,
                    )
                    for color, label in zip(OUTCOME_COLORS, OUTCOME_LABELS)
                ]
                ax.legend(handles=handles, loc="lower left", fontsize=11.0,
                          frameon=True, framealpha=0.92, ncol=2)
    fig.suptitle(f"Commune-level spatial forecasts: target {args.target_date}, H{args.horizon}", fontsize=16.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    if args.output.suffix.lower() != ".png":
        fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
