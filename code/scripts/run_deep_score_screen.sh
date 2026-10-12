#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${DENGUE_PYTHON:-python3}"
export R_LIBS_USER="${DENGUE_R_LIBRARY:-$repo_root/.r-library}"
export PYTHONPATH="$repo_root/code${PYTHONPATH:+:$PYTHONPATH}"
screen_root="${DENGUE_SCORE_SCREEN_ROOT:-$repo_root/results/deep_score_screen}"
draws="${DENGUE_SCREEN_DRAWS:-200}"
mkdir -p "$screen_root"

export_scores() {
  local source_name="$1"
  local feature_mode="$2"
  "$python_bin" "$repo_root/code/scripts/export_deep_scores.py" \
    "$repo_root/results/$source_name/representations.csv" \
    "$repo_root/results/$source_name/model.pt" \
    "$screen_root/$source_name-$feature_mode.csv" \
    --features "$feature_mode"
}

run_one() {
  local source_name="$1"
  local feature_mode="$2"
  local label="$3"
  local output_dir="$screen_root/$label"
  mkdir -p "$output_dir"
  Rscript "$repo_root/code/r/m6_inla.R" \
    "$screen_root/$source_name-$feature_mode.csv" \
    "$output_dir/predictions.csv" \
    "$draws" 1234 3 "$label" 0 5 4 unit \
    >"$output_dir/run.log" 2>&1
}

for source_name in REPR-TCN REPR-GNN REPR-TCN-GNN; do
  export_scores "$source_name" location
  export_scores "$source_name" location_scale
done

run_one REPR-TCN location TCN-SCORE-INLA &
pid_one=$!
run_one REPR-GNN location GNN-SCORE-INLA &
pid_two=$!
run_one REPR-TCN-GNN location TCN-GNN-SCORE-INLA &
pid_three=$!
wait "$pid_one" "$pid_two" "$pid_three"

run_one REPR-TCN location_scale TCN-SCORE-SCALE-INLA &
pid_one=$!
run_one REPR-GNN location_scale GNN-SCORE-SCALE-INLA &
pid_two=$!
run_one REPR-TCN-GNN location_scale TCN-GNN-SCORE-SCALE-INLA &
pid_three=$!
wait "$pid_one" "$pid_two" "$pid_three"
