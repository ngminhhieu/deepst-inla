#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${DENGUE_PYTHON:-python3}"
export R_LIBS_USER="${DENGUE_R_LIBRARY:-$repo_root/.r-library}"
results_root="${DENGUE_RESULTS_ROOT:-$repo_root/results}"
rows_cache="${DENGUE_ROWS_CACHE:-$results_root/inla_rows_cache.rds}"
adjacency="${DENGUE_ADJACENCY_PATH:-$repo_root/code/data/processed/qc_v1/administrative_adjacency.csv}"
input="${DENGUE_REPRESENTATIONS_PATH:-$results_root/REPR-GNN/representations.csv}"
root="${DENGUE_AUGMENTED_FINAL_ROOT:-$results_root/inla_gnn_representation_final}"
peak_root="${DENGUE_PEAK_ROOT:-$results_root/MOE-GNN-HIGH}"
ensemble_root="${DENGUE_ENSEMBLE_ROOT:-$results_root/inla_gnn_representation_peak_exact}"
draws="${DENGUE_FINAL_DRAWS:-500}"
label="INLA-TS-GNN-RES-P4"

run_one() {
  IFS='|' read -r split horizon <<<"$1"
  output="$root/$split/H$horizon"
  mkdir -p "$output"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 Rscript \
    "$repo_root/code/r/representation_augmented_inla.R" \
    "$rows_cache" "$adjacency" "$input" "$output" "$horizon" "$draws" 1234 \
    "$label" latent_residual_pca 4 5 1 "$split" >"$output/run.log" 2>&1
}
export -f run_one
export repo_root rows_cache adjacency input root draws label

printf '%s\n' validation{\|1,\|2,\|3} | xargs -P 3 -I '{}' bash -c 'run_one "$1"' _ '{}'
"$python_bin" "$repo_root/code/scripts/combine_augmented_inla_horizons.py" \
  "$root/validation/INLA-TS-GNN-RES-P4" "$root/validation/H1" \
  "$root/validation/H2" "$root/validation/H3"

printf '%s\n' test{\|1,\|2,\|3} | xargs -P 3 -I '{}' bash -c 'run_one "$1"' _ '{}'
"$python_bin" "$repo_root/code/scripts/combine_augmented_inla_horizons.py" \
  "$root/test/INLA-TS-GNN-RES-P4" "$root/test/H1" "$root/test/H2" "$root/test/H3"

if [[ "${DENGUE_SKIP_LEGACY_GATE:-false}" != "true" ]]; then
  PYTHONPATH="$repo_root/code:$repo_root/code/scripts" "$python_bin" \
    "$repo_root/code/scripts/exact_inla_deep_ensemble.py" \
    "$root/validation/INLA-TS-GNN-RES-P4" \
    "$root/test/INLA-TS-GNN-RES-P4" \
    "$peak_root" \
    "$ensemble_root" --draws "$draws"
fi
