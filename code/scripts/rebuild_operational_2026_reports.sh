#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${DENGUE_PYTHON:-python3}"
raw_dir="$repo_root/code/data/raw/operational_2026"
reference_dir="$repo_root/code/data/reference/operational_2026"
output_dir="$repo_root/outputs"

required=(
  "$raw_dir/rhc_original_mongo_forecasts_2026.json"
  "$reference_dir/rhc_three_method_alerts_2026.xlsx"
  "$output_dir/commune_rerun_db_updated_2026/commune_rhc_alerts_updated_2026-04_to_11.csv"
  "$output_dir/rhc_hybrid_final_2026/data/monthly_rhc_through_2026-08.csv"
  "$output_dir/rhc_hybrid_final_2026/data/rhc_province_lookup.csv"
  "$output_dir/rhc_hybrid_final_2026/models/DEEPST-INLA-RHC-H3/predictions_h3.csv"
  "$output_dir/rhc_hybrid_final_2026/models/DEEPST-INLA-RHC-H3/predictive_samples_h3.npz"
)

for path in "${required[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "Required input is missing: ${path#"$repo_root/"}" >&2
    exit 1
  fi
done

export PYTHONPATH="$repo_root/code:$repo_root/code/scripts${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" "$repo_root/code/scripts/rescore_original_rhc_mongo_with_moh.py" \
  --mongo-export "$raw_dir/rhc_original_mongo_forecasts_2026.json" \
  --observations "$output_dir/rhc_hybrid_final_2026/data/monthly_rhc_through_2026-08.csv" \
  --province-lookup "$output_dir/rhc_hybrid_final_2026/data/rhc_province_lookup.csv" \
  --proposed-predictions "$output_dir/rhc_hybrid_final_2026/models/DEEPST-INLA-RHC-H3/predictions_h3.csv" \
  --proposed-samples "$output_dir/rhc_hybrid_final_2026/models/DEEPST-INLA-RHC-H3/predictive_samples_h3.npz" \
  --output-dir "$output_dir/original_rhc_moh_rescore_2026"

"$python_bin" "$repo_root/code/scripts/build_two_commune_rhc_comparisons.py" \
  --commune "$output_dir/commune_rerun_db_updated_2026/commune_rhc_alerts_updated_2026-04_to_11.csv" \
  --rhc-moh "$output_dir/original_rhc_moh_rescore_2026/original_rhc_predictions_rescored_moh.csv" \
  --operational-alerts "$reference_dir/rhc_three_method_alerts_2026.xlsx" \
  --output-dir "$output_dir/commune_vs_rhc_comparisons_2026"

"$python_bin" "$repo_root/code/scripts/build_native_scale_comparison.py" \
  --commune "$output_dir/commune_rerun_db_updated_2026/commune_rhc_alerts_updated_2026-04_to_11.csv" \
  --rhc "$output_dir/original_rhc_moh_rescore_2026/original_rhc_predictions_rescored_moh.csv" \
  --output-dir "$output_dir/harmonized_alert_comparison_2026"

echo "Rebuilt operational 2026 reporting outputs."
