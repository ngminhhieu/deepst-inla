#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
code_root="$repo_root/code"
python_bin="${DENGUE_PYTHON:-python3}"
rscript_bin="${DENGUE_RSCRIPT:-Rscript}"
config="${DENGUE_CONFIG:-$code_root/config/eval_train2010_2015_valid2016_2020_test2021_2025.yaml}"
run_root="${DENGUE_RESULTS_ROOT:-$repo_root/results/model_runs_train2010_2015_valid2016_2020_test2021_2025}"
final_root="${DENGUE_FINAL_RESULTS_ROOT:-$repo_root/results/final_report_train2010_2015_valid2016_2020_test2021_2025}"
draws="${DENGUE_DRAWS:-500}"
thresholds="${DENGUE_OUTBREAK_THRESHOLDS:-$repo_root/code/data/raw/operational_2026/commune_monthly_thresholds.csv}"
monthly="$code_root/data/processed/feature_suite/monthly_commune_open.csv"
qc_monthly="$code_root/data/processed/qc_v1/monthly_commune.csv"
adjacency="$code_root/data/processed/qc_v1/administrative_adjacency.csv"
source_cache="${DENGUE_SOURCE_ROWS_CACHE:-$repo_root/results/eval_2021_2025_v1/inla_rows_cache.rds}"
rows_cache="$run_root/inla_rows_cache.rds"
features=(t2m_avg rh_avg tp_accum soil_moisture solar_downward dewpoint_c et0_mm)

export PYTHONPATH="$code_root:$code_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export R_LIBS_USER="${DENGUE_R_LIBRARY:-$repo_root/.r-library}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MPLBACKEND=Agg
export DENGUE_TRAIN_START="${DENGUE_TRAIN_START:-2010-01-01}"
export DENGUE_TRAIN_END="${DENGUE_TRAIN_END:-2015-12-01}"
export DENGUE_VALIDATION_START="${DENGUE_VALIDATION_START:-2016-01-01}"
export DENGUE_VALIDATION_END="${DENGUE_VALIDATION_END:-2020-12-01}"
export DENGUE_TEST_START="${DENGUE_TEST_START:-2021-01-01}"
export DENGUE_TEST_END="${DENGUE_TEST_END:-2025-12-01}"
export DENGUE_SPLIT_ON_TARGET_DATE=true
export DENGUE_OUTBREAK_THRESHOLD_COLUMN="${DENGUE_OUTBREAK_THRESHOLD_COLUMN:-epi_threshold}"

mkdir -p "$run_root" "$final_root"

complete_deep() {
  [[ -s "$1/metrics.csv" && -s "$1/predictions.csv" \
    && -s "$1/validation_metrics.csv" && -s "$1/validation_predictions.csv" \
    && -s "$1/predictive_samples.npz" && -s "$1/validation_predictive_samples.npz" ]]
}

run_deep() {
  local label="$1"
  local architecture="$2"
  shift 2
  local output="$run_root/$label"
  if complete_deep "$output"; then
    echo "SKIP completed $label"
    return
  fi
  "$python_bin" "$code_root/scripts/run_experiment.py" \
    --config "$config" --model "$architecture" --output-name "$label" \
    --results-root "$run_root" --monthly-path "$monthly" \
    --climate-columns "${features[@]}" --draws "$draws" \
    --outbreak-thresholds-path "$thresholds" \
    --outbreak-threshold-column "$DENGUE_OUTBREAK_THRESHOLD_COLUMN" \
    --export-validation-predictions "$@"
  if [[ ! -s "$output/validation_metrics.csv" ]]; then
    "$python_bin" "$code_root/scripts/score_saved_validation.py" "$output"
  fi
}

# These definitions are frozen from the current report; only the date split changes.
run_deep TCN-GNN-BASE M5 --export-representations
run_deep LSTM-GNN LSTM-GNN
run_deep DEEPST-ALERT M5 \
  --outbreak-loss-weight 0.25 --outbreak-positive-weight auto \
  --alert-probability-source auxiliary --outbreak-focal-gamma 2

if [[ ! -s "$run_root/TCN-GNN-DEEP-ENSEMBLE/metrics.csv" ]]; then
  "$python_bin" "$code_root/scripts/stack_predictive_distributions.py" \
    --output "$run_root/TCN-GNN-DEEP-ENSEMBLE" \
    --model-label TCN-GNN-DEEP-ENSEMBLE \
    --members "$run_root/TCN-GNN-BASE" "$run_root/LSTM-GNN" \
    --selection-draws 100 --crps-tolerance 0.03 --minimum-precision 0.50
fi

if [[ ! -s "$rows_cache" ]]; then
  DENGUE_RELABEL_SPLITS=true \
    "$rscript_bin" "$code_root/r/filter_inla_rows_cache.R" \
      "$source_cache" "$qc_monthly" "$rows_cache" "$thresholds"
fi

for split in validation test; do
  output="$run_root/INLA-ONLY-$split"
  if [[ -s "$output/INLA-TS/metrics.csv" && -s "$output/INLA-TS/predictions.csv" ]]; then
    echo "SKIP completed INLA-ONLY-$split/INLA-TS"
  else
    mkdir -p "$output"
    DENGUE_EVAL_SPLIT="$split" \
    DENGUE_STATIC_MODELS=INLA-TS \
    DENGUE_SKIP_STATIC_INLA=false \
    DENGUE_RUN_STRONG_INLA=false \
    DENGUE_SAVE_SAMPLES=true \
    DENGUE_ROWS_CACHE="$rows_cache" \
      "$rscript_bin" "$code_root/r/inla_baselines.R" \
        "$qc_monthly" "$adjacency" "$output" "$draws" 1234
  fi
done

for split in validation test; do
  combined="$run_root/REPRESENTATION-INLA/$split/combined"
  if [[ -s "$combined/metrics.csv" && -s "$combined/predictions.csv" ]]; then
    echo "SKIP completed REPRESENTATION-INLA/$split"
    continue
  fi
  for horizon in 1 2 3; do
    output="$run_root/REPRESENTATION-INLA/$split/H$horizon"
    mkdir -p "$output"
    "$rscript_bin" "$code_root/r/representation_augmented_inla.R" \
      "$rows_cache" "$adjacency" "$run_root/TCN-GNN-BASE/representations.csv" \
      "$output" "$horizon" "$draws" 1234 REPRESENTATION-INLA \
      latent_residual_pca 4 5 1 "$split"
  done
  "$python_bin" "$code_root/scripts/combine_augmented_inla_horizons.py" \
    "$combined" \
    "$run_root/REPRESENTATION-INLA/$split/H1" \
    "$run_root/REPRESENTATION-INLA/$split/H2" \
    "$run_root/REPRESENTATION-INLA/$split/H3"
done

if [[ ! -s "$run_root/DEEPST-INLA/metrics.csv" ]]; then
  "$python_bin" "$code_root/scripts/build_multiobjective_ensemble.py" \
    --deep "$run_root/TCN-GNN-BASE" \
    --peak "$run_root/DEEPST-ALERT" \
    --hybrid-validation "$run_root/REPRESENTATION-INLA/validation/combined" \
    --hybrid-test "$run_root/REPRESENTATION-INLA/test/combined" \
    --output "$run_root/DEEPST-INLA" --draws "$draws" --selection-draws 100 \
    --selection-split-label "validation ${DENGUE_VALIDATION_START:0:4}-${DENGUE_VALIDATION_END:0:4}"
fi

"$python_bin" "$code_root/scripts/package_fixed_split_results.py" \
  --run-root "$run_root" --output "$final_root"

"$python_bin" "$code_root/scripts/make_comparison_h3_outbreak_figure.py" \
  --results-root "$run_root" \
  --output "$final_root/Fig_H3_final_model_comparison" \
  --communes 28159,27976,31147,31168 \
  --models \
    inla=INLA-ONLY-test/INLA-TS/predictions.csv \
    base=TCN-GNN-BASE/predictions.csv \
    representation=REPRESENTATION-INLA/test/combined/predictions.csv \
    hybrid=DEEPST-INLA/predictions.csv \
    deep=DEEPST-ALERT/predictions.csv \
  --labels INLA-TS "TCN-GNN Base" Representation-INLA \
    "DeepST-INLA (proposed hybrid)" "DeepST-Alert (proposed deep)" \
  --title "Locked ${DENGUE_TEST_START:0:4}-${DENGUE_TEST_END:0:4} H3 comparison; train ${DENGUE_TRAIN_START:0:4}-${DENGUE_TRAIN_END:0:4}, validation ${DENGUE_VALIDATION_START:0:4}-${DENGUE_VALIDATION_END:0:4}"

"$python_bin" "$code_root/scripts/create_fixed_run_report.py" \
  --run-root "$run_root" --output "$final_root" \
  --train-start "$DENGUE_TRAIN_START" --train-end "$DENGUE_TRAIN_END" \
  --validation-start "$DENGUE_VALIDATION_START" --validation-end "$DENGUE_VALIDATION_END" \
  --test-start "$DENGUE_TEST_START" --test-end "$DENGUE_TEST_END" \
  --thresholds "$thresholds" --threshold-column "$DENGUE_OUTBREAK_THRESHOLD_COLUMN"
echo "Completed fixed-model rerun at $final_root"
