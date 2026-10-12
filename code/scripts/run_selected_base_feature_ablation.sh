#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
code_root="$repo_root/code"
python_bin="${DENGUE_PYTHON:-python3}"
config_path="${DENGUE_CONFIG:-$code_root/config/eval_2021_2025.yaml}"
results_root="${DENGUE_RESULTS_ROOT:-$repo_root/results/feature_ablation_selected_base_2021_2025_v1}"
monthly_path="$code_root/data/processed/feature_suite/monthly_commune_open.csv"
draws="${DENGUE_DRAWS:-500}"

export PYTHONPATH="$code_root:$code_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
mkdir -p "$results_root"

run_feature_set() {
  local label="$1"
  shift
  local output="$results_root/$label"
  if [[ ! -s "$output/metrics.csv" ]]; then
    "$python_bin" "$code_root/scripts/run_experiment.py" \
      --config "$config_path" --model M5 --output-name "$label" \
      --results-root "$results_root" --monthly-path "$monthly_path" \
      --draws "$draws" --export-validation-predictions \
      --climate-columns "$@"
  fi
  if [[ ! -s "$output/validation_metrics.csv" ]]; then
    "$python_bin" "$code_root/scripts/score_saved_validation.py" "$output"
  fi
}

base=(t2m_avg rh_avg tp_accum)
run_feature_set TCN-GNN-BASE "${base[@]}"
run_feature_set TCN-GNN-SOIL "${base[@]}" soil_moisture
run_feature_set TCN-GNN-SOLAR "${base[@]}" solar_downward
run_feature_set TCN-GNN-DEWPOINT "${base[@]}" dewpoint_c
run_feature_set TCN-GNN-ET0 "${base[@]}" et0_mm
run_feature_set TCN-GNN-ALL "${base[@]}" soil_moisture solar_downward dewpoint_c et0_mm

"$python_bin" - "$results_root" <<'PY'
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
metrics = pd.concat(
    [pd.read_csv(path) for path in sorted(root.glob("TCN-GNN-*/validation_metrics.csv"))],
    ignore_index=True,
)
metrics.to_csv(root / "feature_ablation_validation_metrics.csv", index=False)
summary = metrics.groupby("model", as_index=False).agg(
    mean_crps=("crps", "mean"),
    mean_mae=("mae", "mean"),
    mean_brier=("outbreak_brier", "mean"),
    mean_sensitivity=("outbreak_sensitivity", "mean"),
    mean_f1=("outbreak_f1", "mean"),
).sort_values(["mean_crps", "mean_brier", "mean_mae"])
summary.to_csv(root / "feature_ablation_validation_summary.csv", index=False)
print(summary.to_string(index=False))
PY

for output in "$results_root"/TCN-GNN-*; do
  if [[ -d "$output" && -s "$output/metrics.csv" && -s "$output/validation_metrics.csv" ]]; then
    for artifact in model.pt predictive_samples.npz validation_predictive_samples.npz representations.csv predictions.csv validation_predictions.csv; do
      [[ -e "$output/$artifact" ]] && truncate -s 0 "$output/$artifact"
    done
  fi
done

echo "Selected-base feature ablation completed at $results_root"
