#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
code_root="$repo_root/code"
python_bin="${DENGUE_PYTHON:-python3}"
config_path="${DENGUE_CONFIG:-$code_root/config/eval_2021_2025.yaml}"
results_root="${DENGUE_RESULTS_ROOT:-$repo_root/results/feature_tcn_gnn_2021_2025_v1}"
monthly_path="$code_root/data/processed/feature_suite/monthly_commune_open.csv"
draws="${DENGUE_DRAWS:-500}"

export PYTHONPATH="$code_root:$code_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
mkdir -p "$results_root"

run_model() {
  local label="$1"
  shift
  local output="$results_root/$label"
  if [[ ! -s "$output/metrics.csv" ]]; then
    "$python_bin" "$code_root/scripts/run_experiment.py" \
      --config "$config_path" --model M6 --output-name "$label" \
      --results-root "$results_root" --monthly-path "$monthly_path" \
      --draws "$draws" --export-representations --export-validation-predictions \
      --climate-columns "$@"
  fi
  if [[ ! -s "$output/validation_metrics.csv" ]]; then
    "$python_bin" "$code_root/scripts/score_saved_validation.py" "$output"
  fi
}

base=(t2m_avg rh_avg tp_accum)
run_model TCN-GNN-BASE "${base[@]}"
run_model TCN-GNN-SOIL "${base[@]}" soil_moisture
run_model TCN-GNN-SOLAR "${base[@]}" solar_downward
run_model TCN-GNN-DEWPOINT "${base[@]}" dewpoint_c
run_model TCN-GNN-ET0 "${base[@]}" et0_mm
run_model TCN-GNN-ALL "${base[@]}" soil_moisture solar_downward dewpoint_c et0_mm

"$python_bin" - "$results_root" <<'PY'
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("TCN-GNN-*/validation_metrics.csv")):
    frame = pd.read_csv(path)
    rows.append(frame)
metrics = pd.concat(rows, ignore_index=True)
metrics.to_csv(root / "feature_ablation_validation_metrics.csv", index=False)
summary = metrics.groupby("model", as_index=False).agg(
    mean_crps=("crps", "mean"),
    mean_mae=("mae", "mean"),
    mean_brier=("outbreak_brier", "mean"),
    mean_sensitivity=("outbreak_sensitivity", "mean"),
    mean_f1=("outbreak_f1", "mean"),
)
summary = summary.sort_values(["mean_crps", "mean_brier", "mean_mae"])
summary.to_csv(root / "feature_ablation_validation_summary.csv", index=False)
winner = summary.iloc[0]["model"]
(root / "selected_feature_model.txt").write_text(str(winner) + "\n", encoding="utf-8")
print(summary.to_string(index=False))
print(f"Selected by validation mean CRPS: {winner}")
PY

# Feature screening only needs the scored CSVs. Keep the delivery-relevant
# metrics and release large, reproducible training artefacts before the next
# architecture stage.
for label in TCN-GNN-BASE TCN-GNN-SOIL TCN-GNN-SOLAR TCN-GNN-DEWPOINT TCN-GNN-ET0 TCN-GNN-ALL; do
  output="$results_root/$label"
  if [[ -s "$output/metrics.csv" && -s "$output/validation_metrics.csv" ]]; then
    for artifact in model.pt predictive_samples.npz validation_predictive_samples.npz representations.csv predictions.csv validation_predictions.csv; do
      [[ -e "$output/$artifact" ]] && truncate -s 0 "$output/$artifact"
    done
  fi
done

echo "TCN+GNN feature ablation completed at $results_root"
