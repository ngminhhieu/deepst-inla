#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
code_root="$repo_root/code"
python_bin="${DENGUE_PYTHON:-python3}"
config="${DENGUE_CONFIG:-$code_root/config/eval_2021_2025.yaml}"
output="${DENGUE_COMPLETE_UNCERTAINTY_ROOT:-$repo_root/results/complete_uncertainty_2021_2025}"
models="$output/models"
draws="${DENGUE_DRAWS:-500}"
monthly="$code_root/data/processed/feature_suite/monthly_commune_open.csv"
features=(t2m_avg rh_avg tp_accum soil_moisture solar_downward dewpoint_c et0_mm)

mkdir -p "$models"
export PYTHONPATH="$code_root:$code_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec > >(tee -a "$output/neural.log") 2>&1

complete_deep() {
  [[ -s "$1/metrics.csv" && -s "$1/predictions.csv" \
    && -s "$1/predictive_samples.npz" \
    && -s "$1/validation_predictions.csv" \
    && -s "$1/validation_predictive_samples.npz" ]]
}

run_deep() {
  local label="$1"
  local architecture="$2"
  shift 2
  if complete_deep "$models/$label"; then
    echo "SKIP completed $label"
    return
  fi
  echo "RUN $label"
  "$python_bin" "$code_root/scripts/run_experiment.py" \
    --config "$config" --model "$architecture" --output-name "$label" \
    --results-root "$models" --monthly-path "$monthly" \
    --climate-columns "${features[@]}" --draws "$draws" \
    --export-validation-predictions "$@"
  "$python_bin" "$code_root/scripts/score_saved_validation.py" "$models/$label"
  echo "DONE $label"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START main-model neural components"
run_deep TCN-GNN-BASE M5 --export-representations
run_deep LSTM-GNN LSTM-GNN
run_deep DEEPST-ALERT M5 \
  --outbreak-loss-weight 0.25 --outbreak-positive-weight auto \
  --alert-probability-source auxiliary --outbreak-focal-gamma 2

if [[ ! -s "$models/TCN-GNN-DEEP-ENSEMBLE/metrics.csv" ]]; then
  "$python_bin" "$code_root/scripts/stack_predictive_distributions.py" \
    --output "$models/TCN-GNN-DEEP-ENSEMBLE" \
    --model-label TCN-GNN-DEEP-ENSEMBLE \
    --members "$models/TCN-GNN-BASE" "$models/LSTM-GNN" \
    --selection-draws 100 --crps-tolerance 0.03 --minimum-precision 0.50
fi
date -Iseconds > "$output/NEURAL_COMPLETE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] COMPLETE main-model neural components"
