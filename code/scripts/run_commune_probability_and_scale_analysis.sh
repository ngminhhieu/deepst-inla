#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
archive="${DENGUE_ARCHIVE_ROOT:-}"
source_root="${DENGUE_SOURCE_ROOT:-$repo_root/../dengue_forecast_v1/dengue_training}"
python_bin="${DENGUE_PYTHON:-python3}"

if [[ -z "$archive" ]]; then
  echo "Set DENGUE_ARCHIVE_ROOT to the recovered fixed-split result directory." >&2
  exit 2
fi

common="$archive"
"$python_bin" "$repo_root/code/paper_pipeline/export_commune_probabilities.py" \
  --model "DeepST-INLA=$common/DEEPST-INLA/validation_predictions.csv=$common/DEEPST-INLA/predictions.csv" \
  --model "TCN-GNN=$common/TCN-GNN-BASE/validation_predictions.csv=$common/TCN-GNN-BASE/predictions.csv" \
  --model "DeepST-Alert=$common/DEEPST-ALERT/validation_predictions.csv=$common/DEEPST-ALERT/predictions.csv" \
  --model "INLA-TS=$common/INLA-ONLY-validation/INLA-TS/predictions.csv=$common/INLA-ONLY-test/INLA-TS/predictions.csv" \
  --output "$repo_root/results/commune_probability_export"

"$python_bin" "$repo_root/code/paper_pipeline/analyze_spatial_support.py" \
  --predictions "$common/TCN-GNN-BASE/predictions.csv" \
  --samples "$common/TCN-GNN-BASE/predictive_samples.npz" \
  --validation-predictions "$common/TCN-GNN-BASE/validation_predictions.csv" \
  --validation-samples "$common/TCN-GNN-BASE/validation_predictive_samples.npz" \
  --monthly "$source_root/data/processed/monthly_commune_from_mongodb.csv" \
  --centroids "$source_root/data/static/inla/commune_centroids.csv" \
  --adjacency "$source_root/data/static/inla/administrative_adjacency.csv" \
  --settings "$source_root/config.yaml" \
  --output "$repo_root/results/spatial_support"
