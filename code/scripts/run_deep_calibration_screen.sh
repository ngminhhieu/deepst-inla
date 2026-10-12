#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${DENGUE_PYTHON:-python3}"
export PYTHONPATH="$repo_root/code${PYTHONPATH:+:$PYTHONPATH}"
feature_root="$repo_root/results/deep_score_screen"
output_root="${DENGUE_CALIBRATION_ROOT:-$repo_root/results/deep_calibration_screen}"

run_one() {
  local source="$1"
  local weight="$2"
  local label="$3"
  "$python_bin" "$repo_root/code/scripts/calibrate_deep_outputs.py" \
    "$feature_root/$source-location_scale.csv" \
    "$output_root/$label" \
    --model-label "$label" \
    --outbreak-weight "$weight"
}

run_one REPR-GNN 0 GNN-CAL &
pid_one=$!
run_one REPR-GNN 1 GNN-CAL-W1 &
pid_two=$!
run_one REPR-GNN 3 GNN-CAL-W3 &
pid_three=$!
wait "$pid_one" "$pid_two" "$pid_three"

"$python_bin" "$repo_root/code/scripts/calibrate_deep_outputs.py" \
  "$feature_root/REPR-GNN-location_scale.csv" \
  "$output_root/GNN-CAL-FIT" \
  --model-label GNN-CAL-FIT --calibration-split fit &
pid_one=$!
"$python_bin" "$repo_root/code/scripts/calibrate_deep_outputs.py" \
  "$feature_root/REPR-TCN-GNN-location_scale.csv" \
  "$output_root/TCN-GNN-CAL-FIT" \
  --model-label TCN-GNN-CAL-FIT --calibration-split fit &
pid_two=$!
wait "$pid_one" "$pid_two"

run_one REPR-TCN-GNN 0 TCN-GNN-CAL &
pid_one=$!
run_one REPR-TCN-GNN 1 TCN-GNN-CAL-W1 &
pid_two=$!
run_one REPR-TCN-GNN 3 TCN-GNN-CAL-W3 &
pid_three=$!
wait "$pid_one" "$pid_two" "$pid_three"
