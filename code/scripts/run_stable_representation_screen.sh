#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export R_LIBS_USER="${DENGUE_R_LIBRARY:-$repo_root/.r-library}"
screen_root="${DENGUE_SCREEN_ROOT:-$repo_root/results/stable_representation_screen}"
draws="${DENGUE_SCREEN_DRAWS:-200}"
seed="${DENGUE_SCREEN_SEED:-1234}"
clip_sd="${DENGUE_SCREEN_CLIP_SD:-5}"
fixed_precision="${DENGUE_SCREEN_FIXED_PRECISION:-4}"
mkdir -p "$screen_root"

run_one() {
  local source_name="$1"
  local components="$2"
  local label="$3"
  local output_dir="$screen_root/$label"
  mkdir -p "$output_dir"
  Rscript "$repo_root/code/r/m6_inla.R" \
    "$repo_root/results/$source_name/representations.csv" \
    "$output_dir/predictions.csv" \
    "$draws" "$seed" 3 "$label" "$components" "$clip_sd" "$fixed_precision" \
    >"$output_dir/run.log" 2>&1
}

# Run three candidates at a time to reduce wall-clock time without crowding INLA.
run_one REPR-TCN 4 TCN-PCA4-INLA &
pid_one=$!
run_one REPR-TCN 8 TCN-PCA8-INLA &
pid_two=$!
run_one REPR-TCN 12 TCN-PCA12-INLA &
pid_three=$!
wait "$pid_one" "$pid_two" "$pid_three"
run_one REPR-TCN-GNN 4 TCN-GNN-PCA4-INLA &
pid_one=$!
run_one REPR-TCN-GNN 8 TCN-GNN-PCA8-INLA &
pid_two=$!
run_one REPR-TCN-GNN 12 TCN-GNN-PCA12-INLA &
pid_three=$!
wait "$pid_one" "$pid_two" "$pid_three"
