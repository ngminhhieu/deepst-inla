#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
code_root="$repo_root/code"
python_bin="${DENGUE_PYTHON:-python3}"
config="${DENGUE_CONFIG:-$code_root/config/eval_2021_2025.yaml}"
results_root="${DENGUE_RESULTS_ROOT:-$repo_root/results/multitask_warning_2021_2025_v1}"
monthly="$code_root/data/processed/feature_suite/monthly_commune_open.csv"
features=(t2m_avg rh_avg tp_accum soil_moisture solar_downward dewpoint_c et0_mm)

export PYTHONPATH="$code_root:$code_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
mkdir -p "$results_root"

run_warning() {
  local label="$1"
  shift
  output="$results_root/$label"
  if [[ ! -s "$output/metrics.csv" ]]; then
    "$python_bin" "$code_root/scripts/run_experiment.py" \
      --config "$config" --model M5 --output-name "$label" \
      --results-root "$results_root" --monthly-path "$monthly" \
      --climate-columns "${features[@]}" --draws "${DENGUE_DRAWS:-500}" \
      --export-validation-predictions \
      --outbreak-loss-weight 0.25 --outbreak-positive-weight auto \
      --alert-probability-source auxiliary "$@"
  fi
  if [[ ! -s "$output/validation_metrics.csv" ]]; then
    "$python_bin" "$code_root/scripts/score_saved_validation.py" "$output"
  fi
}

run_warning ALERT-WEIGHTED-BCE --outbreak-focal-gamma 0
run_warning ALERT-FOCAL-BCE --outbreak-focal-gamma 2
run_warning ALERT-PEAK-FOCAL --peak-loss-weight 1 --outbreak-focal-gamma 2

"$python_bin" - "$results_root" <<'PY'
from pathlib import Path
import sys
import numpy as np
import pandas as pd

root = Path(sys.argv[1])
labels = ("ALERT-WEIGHTED-BCE", "ALERT-FOCAL-BCE", "ALERT-PEAK-FOCAL")
metrics = pd.concat([pd.read_csv(root / label / "validation_metrics.csv") for label in labels])
summary = metrics.groupby("model", as_index=False).agg(
    mean_crps=("crps", "mean"), mean_brier=("outbreak_brier", "mean"),
    mean_precision=("outbreak_precision", "mean"),
    mean_specificity=("outbreak_specificity", "mean"),
    mean_sensitivity=("outbreak_sensitivity", "mean"), mean_f1=("outbreak_f1", "mean"),
)
summary["warning_objective"] = np.power(
    summary["mean_precision"] * summary["mean_specificity"]
    * summary["mean_sensitivity"] * summary["mean_f1"]
    * (1 - summary["mean_brier"]), 1 / 5
)
summary = summary.sort_values(["warning_objective", "mean_brier"], ascending=[False, True])
summary.to_csv(root / "warning_validation_summary.csv", index=False)
winner = str(summary.iloc[0]["model"])
(root / "selected_warning_model.txt").write_text(winner + "\n", encoding="utf-8")
print(summary.to_string(index=False))
print("Selected warning expert:", winner)
PY
