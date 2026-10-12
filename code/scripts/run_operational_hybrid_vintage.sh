#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: run_operational_hybrid_vintage.sh ISSUE_DATE MONTHLY_PANEL OUTPUT_ROOT" >&2
  exit 2
fi

issue_date="$1"
monthly_panel="$2"
output_root="$3"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
code_root="$repo_root/code"
python_bin="${DENGUE_PYTHON:-python3}"
draws="${DENGUE_DRAWS:-500}"
epochs="${DENGUE_EPOCHS:-33}"
selection="${DENGUE_SELECTION:-$repo_root/results/uncertainty_complete/model_metrics/DEEPST-INLA/selection.json}"

mkdir -p "$output_root/runtime" "$output_root/model" "$output_root/work"
# Keep intermediate components so a failed INLA phase can resume without
# refitting both deep models.
tmp_dir="$output_root/work"

export PYTHONPATH="$code_root:$code_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
# Use the caller's R library by default.  Set DENGUE_R_LIBRARY when a pinned
# R-INLA installation is required on the execution host.
if [[ -n "${DENGUE_R_LIBRARY:-}" ]]; then
  export R_LIBS_USER="$DENGUE_R_LIBRARY"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export DENGUE_INLA_THREADS="${DENGUE_INLA_THREADS:-2:1}"
export MPLBACKEND=Agg

if [[ ! -s "$tmp_dir/deep_fit_manifest.json" || \
      ! -s "$tmp_dir/deep/predictive_samples.npz" || \
      ! -s "$tmp_dir/peak/predictive_samples.npz" ]]; then
  "$python_bin" "$code_root/scripts/fit_rolling_refit_deep_vintage.py" \
    --config "$code_root/config/eval_cv_retrain_2021_2025.yaml" \
    --issue-date "$issue_date" --output "$tmp_dir" --epochs "$epochs" \
    --draws "$draws" --device cpu --monthly-data "$monthly_panel" \
    --climate-columns t2m_avg rh_avg tp_accum soil_moisture solar_downward dewpoint_c et0_mm
fi

target_end="$("$python_bin" -c 'import pandas as pd,sys; print((pd.Timestamp(sys.argv[1])+pd.DateOffset(months=3)).date())' "$issue_date")"
source_cache="$tmp_dir/source_rows.rds"
if [[ ! -s "$source_cache" ]]; then
  DENGUE_ROWS_CACHE="$source_cache" DENGUE_CACHE_ONLY=true \
    DENGUE_TEST_END="$target_end" DENGUE_SPLIT_ON_TARGET_DATE=true \
    /usr/bin/Rscript "$code_root/r/inla_baselines.R" \
      "$monthly_panel" "$code_root/data/processed/qc_v1/administrative_adjacency.csv" \
      "$tmp_dir/cache_only" "$draws" 1234
fi

if [[ ! -s "$tmp_dir/rolling_rows.rds" ]]; then
  /usr/bin/Rscript "$code_root/r/make_rolling_refit_rows_cache.R" \
    "$source_cache" "$issue_date" "$tmp_dir/rolling_rows.rds"
fi

for horizon in 1 2 3; do
  mkdir -p "$tmp_dir/hybrid/H$horizon"
  if [[ -s "$tmp_dir/hybrid/H$horizon/predictions.csv" && \
        -s "$tmp_dir/hybrid/H$horizon/samples_H${horizon}.bin" ]]; then
    continue
  fi
  /usr/bin/Rscript "$code_root/r/representation_augmented_inla.R" \
    "$tmp_dir/rolling_rows.rds" \
    "$code_root/data/processed/qc_v1/administrative_adjacency.csv" \
    "$tmp_dir/deep/representations.csv" "$tmp_dir/hybrid/H$horizon" \
    "$horizon" "$draws" 1234 REPRESENTATION-INLA \
    latent_residual_pca 4 5 1 test
done

"$python_bin" "$code_root/scripts/combine_augmented_inla_horizons.py" \
  "$tmp_dir/hybrid/combined" "$tmp_dir/hybrid/H1" \
  "$tmp_dir/hybrid/H2" "$tmp_dir/hybrid/H3"
"$python_bin" "$code_root/scripts/apply_locked_multiobjective_policy.py" \
  --deep "$tmp_dir/deep" --peak "$tmp_dir/peak" \
  --hybrid "$tmp_dir/hybrid/combined" --selection "$selection" \
  --output "$output_root/model" --draws "$draws"

cp "$tmp_dir/deep_fit_manifest.json" "$output_root/model/"
echo "DONE $issue_date -> $output_root/model"
