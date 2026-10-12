#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
code_root="$repo_root/code"
python_bin="${DENGUE_PYTHON:-python3}"
rscript_bin="${DENGUE_RSCRIPT:-Rscript}"
root="${DENGUE_COMPLETE_UNCERTAINTY_ROOT:-$repo_root/results/complete_uncertainty_2021_2025}"
models="$root/models"
draws="${DENGUE_DRAWS:-500}"
qc_monthly="$code_root/data/processed/qc_v1/monthly_commune.csv"
adjacency="$code_root/data/processed/qc_v1/administrative_adjacency.csv"
source_cache="${DENGUE_SOURCE_ROWS_CACHE:-$repo_root/results/optional_analyses/input_audit/inla_rows_cache.rds}"
rows_cache="$root/inla_rows_cache.rds"

export PYTHONPATH="$code_root:$code_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export R_LIBS_USER="${DENGUE_R_LIBRARY:-$repo_root/.r-library}"
export DENGUE_TRAIN_START=2010-01-01
export DENGUE_TRAIN_END=2018-12-01
export DENGUE_VALIDATION_START=2019-01-01
export DENGUE_VALIDATION_END=2020-12-01
export DENGUE_TEST_START=2021-01-01
export DENGUE_TEST_END=2025-12-01
export DENGUE_SPLIT_ON_TARGET_DATE=true
export DENGUE_INLA_THREADS="${DENGUE_INLA_THREADS:-8:1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

mkdir -p "$root" "$models" "$root/final"
exec > >(tee -a "$root/hybrid.log") 2>&1

while [[ ! -s "$root/NEURAL_COMPLETE" || ! -s "$root/INLA_READY" ]]; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] waiting for neural/INLA readiness"
  sleep 30
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] START Bayesian and hybrid components"

if [[ ! -s "$rows_cache" ]]; then
  DENGUE_RELABEL_SPLITS=true "$rscript_bin" \
    "$code_root/r/filter_inla_rows_cache.R" "$source_cache" "$qc_monthly" "$rows_cache"
fi

for split in validation test; do
  output="$models/INLA-ONLY-$split"
  if [[ ! -s "$output/INLA-TS/metrics.csv" || ! -s "$output/INLA-TS/samples_H3.bin" ]]; then
    mkdir -p "$output"
    DENGUE_EVAL_SPLIT="$split" DENGUE_STATIC_MODELS=INLA-TS \
    DENGUE_SKIP_STATIC_INLA=false DENGUE_RUN_STRONG_INLA=false \
    DENGUE_SAVE_SAMPLES=true DENGUE_ROWS_CACHE="$rows_cache" \
      "$rscript_bin" "$code_root/r/inla_baselines.R" \
        "$qc_monthly" "$adjacency" "$output" "$draws" 1234
  fi
done

for split in validation test; do
  combined="$models/REPRESENTATION-INLA/$split/combined"
  if [[ -s "$combined/metrics.csv" && -s "$combined/samples_H3.bin" ]]; then
    continue
  fi
  for horizon in 1 2 3; do
    output="$models/REPRESENTATION-INLA/$split/H$horizon"
    mkdir -p "$output"
    if [[ ! -s "$output/metrics.csv" || ! -s "$output/samples_H${horizon}.bin" ]]; then
      "$rscript_bin" "$code_root/r/representation_augmented_inla.R" \
        "$rows_cache" "$adjacency" "$models/TCN-GNN-BASE/representations.csv" \
        "$output" "$horizon" "$draws" 1234 REPRESENTATION-INLA \
        latent_residual_pca 4 5 1 "$split"
    fi
  done
  "$python_bin" "$code_root/scripts/combine_augmented_inla_horizons.py" \
    "$combined" "$models/REPRESENTATION-INLA/$split/H1" \
    "$models/REPRESENTATION-INLA/$split/H2" "$models/REPRESENTATION-INLA/$split/H3"
done

if [[ ! -s "$models/DEEPST-INLA/metrics.csv" || ! -s "$models/DEEPST-INLA/predictive_samples.npz" ]]; then
  "$python_bin" "$code_root/scripts/build_multiobjective_ensemble.py" \
    --deep "$models/TCN-GNN-BASE" --peak "$models/DEEPST-ALERT" \
    --hybrid-validation "$models/REPRESENTATION-INLA/validation/combined" \
    --hybrid-test "$models/REPRESENTATION-INLA/test/combined" \
    --output "$models/DEEPST-INLA" --draws "$draws" --selection-draws 100 \
    --selection-split-label "validation 2019-2020"
fi

"$python_bin" "$code_root/scripts/package_fixed_split_results.py" \
  --run-root "$models" --output "$root/final"

"$python_bin" "$code_root/paper_pipeline/bootstrap_all_models.py" \
  --output "$root/bootstrap" --reps 2000 --reference DeepST-INLA \
  --model "INLA-TS=$models/INLA-ONLY-test/INLA-TS" \
  --model "TCN-GNN-Base=$models/TCN-GNN-BASE" \
  --model "LSTM-GNN=$models/LSTM-GNN" \
  --model "DeepST-Alert=$models/DEEPST-ALERT" \
  --model "TCN-GNN-Deep-Ensemble=$models/TCN-GNN-DEEP-ENSEMBLE" \
  --model "Representation-INLA=$models/REPRESENTATION-INLA/test/combined" \
  --model "DeepST-INLA=$models/DEEPST-INLA"
date -Iseconds > "$root/HYBRID_COMPLETE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] COMPLETE Bayesian and hybrid components"
