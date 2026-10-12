#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: run_one_rolling_refit_vintage.sh ISSUE_DATE OUTPUT_ROOT" >&2
  exit 2
fi
issue_date="$1"
output_root="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
code_root="$repo_root/code"
python_bin="${DENGUE_PYTHON:-python3}"
final_dir="$output_root/vintages/${issue_date//-/}/final"
if [[ -s "$final_dir/predictions.csv" && -s "$final_dir/predictive_samples.npz" ]]; then
  echo "SKIP $issue_date"
  exit 0
fi

runtime_root="$output_root/runtime"
mkdir -p "$runtime_root" "$output_root/vintages/${issue_date//-/}"
tmp_dir="$(mktemp -d "$runtime_root/${issue_date//-/}.XXXXXX")"
cleanup() { rm -rf -- "$tmp_dir"; }
trap cleanup EXIT

export PYTHONPATH="$code_root:$code_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export R_LIBS_USER="${DENGUE_R_LIBRARY:-$repo_root/.r-library}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export DENGUE_INLA_THREADS="${DENGUE_INLA_THREADS:-2:1}"
export MPLBACKEND=Agg

"$python_bin" "$code_root/scripts/fit_rolling_refit_deep_vintage.py" \
  --config "$code_root/config/eval_cv_retrain_2021_2025.yaml" \
  --issue-date "$issue_date" --output "$tmp_dir" --epochs 33 --draws 500 --device cpu \
  --monthly-data "$code_root/data/processed/feature_suite/monthly_commune_open.csv" \
  --climate-columns t2m_avg rh_avg tp_accum soil_moisture solar_downward dewpoint_c et0_mm

rolling_rows_cache="${DENGUE_ROLLING_ROWS_CACHE:-$repo_root/results/eval_2021_2025_v1/inla_rows_cache.rds}"
/usr/bin/Rscript "$code_root/r/make_rolling_refit_rows_cache.R" \
  "$rolling_rows_cache" \
  "$issue_date" "$tmp_dir/rolling_rows.rds"

for horizon in 1 2 3; do
  mkdir -p "$tmp_dir/hybrid/H$horizon"
  /usr/bin/Rscript "$code_root/r/representation_augmented_inla.R" \
    "$tmp_dir/rolling_rows.rds" \
    "$code_root/data/processed/qc_v1/administrative_adjacency.csv" \
    "$tmp_dir/deep/representations.csv" "$tmp_dir/hybrid/H$horizon" \
    "$horizon" 500 1234 REPRESENTATION-INLA \
    latent_residual_pca 4 5 1 test
done
"$python_bin" "$code_root/scripts/combine_augmented_inla_horizons.py" \
  "$tmp_dir/hybrid/combined" "$tmp_dir/hybrid/H1" "$tmp_dir/hybrid/H2" "$tmp_dir/hybrid/H3"
"$python_bin" "$code_root/scripts/apply_locked_multiobjective_policy.py" \
  --deep "$tmp_dir/deep" --peak "$tmp_dir/peak" --hybrid "$tmp_dir/hybrid/combined" \
  --selection "$repo_root/results/final_multiobjective_model_2021_2025/selection.json" \
  --output "$final_dir" --draws 500
cp "$tmp_dir/deep_fit_manifest.json" "$output_root/vintages/${issue_date//-/}/"
echo "DONE $issue_date"
