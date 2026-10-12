#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
code_root="$repo_root/code"
results_root="$repo_root/results/province_scale"
python_bin="${DENGUE_PYTHON:-python3}"
rscript_bin="${DENGUE_RSCRIPT:-Rscript}"
draws="${DENGUE_DRAWS:-500}"

: "${DENGUE_PROVINCE_SOURCE:?Set DENGUE_PROVINCE_SOURCE to the intermediate-unit monthly CSV}"
: "${DENGUE_COMMUNE_REFERENCE:?Set DENGUE_COMMUNE_REFERENCE to the commune monthly CSV}"
: "${DENGUE_COMMUNE_CENTROIDS:?Set DENGUE_COMMUNE_CENTROIDS to commune centroids CSV}"
: "${DENGUE_COMMUNE_ADJACENCY:?Set DENGUE_COMMUNE_ADJACENCY to commune adjacency CSV}"

export PYTHONPATH="$code_root:$code_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export R_LIBS_USER="${DENGUE_R_LIBRARY:-$repo_root/.r-library}"
export DENGUE_INLA_THREADS="${DENGUE_INLA_THREADS:-4:1}"
export DENGUE_TRAIN_START=2010-01-01
export DENGUE_TRAIN_END=2018-12-01
export DENGUE_VALIDATION_START=2019-01-01
export DENGUE_VALIDATION_END=2020-12-01
export DENGUE_TEST_START=2021-01-01
export DENGUE_TEST_END=2025-12-01
export DENGUE_SPLIT_ON_TARGET_DATE=true

"$python_bin" "$code_root/paper_pipeline/prepare_province_scale.py" \
  --source "$DENGUE_PROVINCE_SOURCE" \
  --commune-monthly "$DENGUE_COMMUNE_REFERENCE" \
  --commune-centroids "$DENGUE_COMMUNE_CENTROIDS" \
  --commune-adjacency "$DENGUE_COMMUNE_ADJACENCY" \
  --output-dir "$results_root/data"

run_deep() {
  local label="$1"
  shift
  # Phase 1 produces genuinely held-out validation forecasts for policy selection.
  "$python_bin" "$code_root/scripts/run_experiment.py" \
    --config "$code_root/config/eval_province_2021_2025.yaml" \
    --model M6 --output-name "$label-selection" --results-root "$results_root/models" \
    --export-representations --export-validation-predictions --draws "$draws" \
    --refit-on-train-validation false "$@"
  # Phase 2 uses the selected epoch count and refits on the complete development period.
  "$python_bin" "$code_root/scripts/run_experiment.py" \
    --config "$code_root/config/eval_province_2021_2025.yaml" \
    --model M6 --output-name "$label" --results-root "$results_root/models" \
    --export-representations --export-validation-predictions --draws "$draws" \
    --refit-on-train-validation true "$@"
  cp "$results_root/models/$label-selection/validation_predictions.csv" \
    "$results_root/models/$label/validation_predictions.csv"
  cp "$results_root/models/$label-selection/validation_predictive_samples.npz" \
    "$results_root/models/$label/validation_predictive_samples.npz"
}
run_deep TCN-GNN
run_deep DeepST-Alert --outbreak-loss-weight 1 --outbreak-positive-weight auto \
  --outbreak-focal-gamma 2 --alert-probability-source auxiliary

rows_cache="$results_root/models/inla_rows_cache.rds"
for split in validation test; do
  DENGUE_EVAL_SPLIT="$split" DENGUE_STATIC_MODELS=INLA-TS \
  DENGUE_SAVE_SAMPLES=true DENGUE_ROWS_CACHE="$rows_cache" \
    "$rscript_bin" "$code_root/r/inla_baselines.R" \
      "$results_root/data/monthly_province.csv" \
      "$results_root/data/province_adjacency.csv" \
      "$results_root/models/INLA-$split" "$draws" 1234
done

for split in validation test; do
  horizon_dirs=()
  for horizon in 1 2 3; do
    output="$results_root/models/Representation-INLA/$split/H$horizon"
    mkdir -p "$output"
    representation="$results_root/models/TCN-GNN/representations.csv"
    if [[ "$split" == validation ]]; then
      representation="$results_root/models/TCN-GNN-selection/representations.csv"
    fi
    "$rscript_bin" "$code_root/r/representation_augmented_inla.R" \
      "$rows_cache" "$results_root/data/province_adjacency.csv" \
      "$representation" "$output" \
      "$horizon" "$draws" 1234 Representation-INLA latent_residual_pca 4 5 1 "$split"
    horizon_dirs+=("$output")
  done
  "$python_bin" "$code_root/scripts/combine_augmented_inla_horizons.py" \
    "$results_root/models/Representation-INLA/$split/combined" "${horizon_dirs[@]}"
done

"$python_bin" "$code_root/scripts/build_multiobjective_ensemble.py" \
  --deep "$results_root/models/TCN-GNN" \
  --peak "$results_root/models/DeepST-Alert" \
  --hybrid-validation "$results_root/models/Representation-INLA/validation/combined" \
  --hybrid-test "$results_root/models/Representation-INLA/test/combined" \
  --output "$results_root/models/DeepST-INLA" --draws "$draws" \
  --selection-draws 100 --selection-split-label "validation 2019-2020"

"$python_bin" "$code_root/paper_pipeline/analyze_province_scale.py" \
  --models-root "$results_root/models" \
  --commune-metrics "$repo_root/results/current/final_metrics_mean_H1_H3.csv" \
  --output-dir "$results_root"
