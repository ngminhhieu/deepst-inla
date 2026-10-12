#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export R_LIBS_USER="${DENGUE_R_LIBRARY:-$repo_root/.r-library}"
score_root="$repo_root/results/deep_score_screen"
screen_root="${DENGUE_GAUSSIAN_SCREEN_ROOT:-$repo_root/results/gaussian_representation_screen}"
draws="${DENGUE_SCREEN_DRAWS:-200}"
mkdir -p "$screen_root"

run_score() {
  local input_name="$1"
  local label="$2"
  local output_dir="$screen_root/$label"
  mkdir -p "$output_dir"
  Rscript "$repo_root/code/r/m6_inla.R" \
    "$score_root/$input_name.csv" "$output_dir/predictions.csv" \
    "$draws" 1234 3 "$label" 0 5 4 unit gaussian_log1p \
    >"$output_dir/run.log" 2>&1
}

run_score REPR-GNN-location GNN-SCORE-GAUSS-INLA &
pid_one=$!
run_score REPR-GNN-location_scale GNN-SCORE-SCALE-GAUSS-INLA &
pid_two=$!
run_score REPR-TCN-GNN-location TCN-GNN-SCORE-GAUSS-INLA &
pid_three=$!
run_score REPR-TCN-GNN-location_scale TCN-GNN-SCORE-SCALE-GAUSS-INLA &
pid_four=$!
wait "$pid_one" "$pid_two" "$pid_three" "$pid_four"
