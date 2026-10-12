#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

required=(
  "code/data/raw/operational_2026/mongo_commune_2026.json"
  "code/data/raw/operational_2026/rhc_original_mongo_forecasts_2026.json"
  "code/data/raw/operational_2026/rhc_monthly_2010_to_2026-09.csv"
  "code/data/raw/operational_2026/commune_monthly_thresholds.csv"
  "code/data/raw/operational_2026/original_commune_forecast_quantiles.csv"
  "code/data/reference/operational_2026/commune_moh_thresholds_2015_2026.csv"
  "code/data/reference/operational_2026/rhc_three_method_alerts_2026.xlsx"
)

missing=0
for relative_path in "${required[@]}"; do
  if [[ ! -s "$repo_root/$relative_path" ]]; then
    echo "MISSING $relative_path" >&2
    missing=1
  else
    echo "OK      $relative_path"
  fi
done

if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

echo "Operational 2026 input layout is complete."
