#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
code_root="$repo_root/code"
python_bin="${DENGUE_PYTHON:-python3}"
rscript_bin="${DENGUE_RSCRIPT:-Rscript}"
config_path="${DENGUE_CONFIG:-$code_root/config/eval_2021_2025.yaml}"
results_root="${DENGUE_RESULTS_ROOT:-$repo_root/results/eval_2021_2025_v1}"
draws="${DENGUE_DRAWS:-500}"
monthly_path="$code_root/data/processed/qc_v1/monthly_commune.csv"
open_panel="$code_root/data/processed/feature_suite/monthly_commune_open.csv"
adjacency_path="$code_root/data/processed/qc_v1/administrative_adjacency.csv"
rows_cache="$results_root/inla_rows_cache.rds"
source_rows_cache="${DENGUE_SOURCE_ROWS_CACHE:-$repo_root/results/qc_v1/inla_rows_cache.rds}"

export PYTHONPATH="$code_root:$code_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export R_LIBS_USER="${DENGUE_R_LIBRARY:-$repo_root/.r-library}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MPLBACKEND=Agg
export DENGUE_TRAIN_START="${DENGUE_TRAIN_START:-2010-01-01}"
export DENGUE_TRAIN_END="${DENGUE_TRAIN_END:-2018-12-01}"
export DENGUE_VALIDATION_START="${DENGUE_VALIDATION_START:-2019-01-01}"
export DENGUE_VALIDATION_END="${DENGUE_VALIDATION_END:-2020-12-01}"
export DENGUE_TEST_START="${DENGUE_TEST_START:-2021-01-01}"
export DENGUE_TEST_END="${DENGUE_TEST_END:-2025-12-01}"
export DENGUE_SPLIT_ON_TARGET_DATE="${DENGUE_SPLIT_ON_TARGET_DATE:-true}"

mkdir -p "$results_root" "$repo_root/logs"

complete_deep() {
  [[ -s "$1/metrics.csv" && -s "$1/predictions.csv" && -s "$1/representations.csv" ]]
}

run_deep() {
  local label="$1"
  shift
  if complete_deep "$results_root/$label"; then
    echo "SKIP completed $label"
    return
  fi
  "$python_bin" "$code_root/scripts/run_experiment.py" \
    --config "$config_path" --model GNN --output-name "$label" \
    --results-root "$results_root" --draws "$draws" \
    --export-representations --export-validation-predictions "$@"
}

run_deep DEEP-LN-GNN
run_deep DEEP-NB-GNN --likelihood negative_binomial
run_deep DEEP-LAND-GNN \
  --monthly-path "$open_panel" \
  --climate-columns t2m_avg rh_avg tp_accum soil_moisture solar_downward dewpoint_c et0_mm
run_deep DEEP-PEAK-GNN --peak-loss-weight 1

if [[ ! -s "$rows_cache" && -s "$source_rows_cache" ]]; then
  DENGUE_RELABEL_SPLITS=true \
    "$rscript_bin" "$code_root/r/filter_inla_rows_cache.R" \
      "$source_rows_cache" "$monthly_path" "$rows_cache"
fi

for split in validation test; do
  output="$results_root/INLA-ONLY-$split"
  if [[ -s "$output/INLA-TS/metrics.csv" && -s "$output/INLA-TS/predictions.csv" ]]; then
    echo "SKIP completed INLA-ONLY-$split"
    continue
  fi
  mkdir -p "$output"
  DENGUE_EVAL_SPLIT="$split" \
  DENGUE_STATIC_MODELS=INLA-TS \
  DENGUE_SKIP_STATIC_INLA=false \
  DENGUE_RUN_STRONG_INLA=false \
  DENGUE_SAVE_SAMPLES=true \
  DENGUE_ROWS_CACHE="$rows_cache" \
    "$rscript_bin" "$code_root/r/inla_baselines.R" \
      "$monthly_path" "$adjacency_path" "$output" "$draws" 1234
done

run_hybrid() {
  local deep_label="$1"
  local hybrid_label="$2"
  local representation="$results_root/$deep_label/representations.csv"
  for split in validation test; do
    local combined="$results_root/$hybrid_label/$split/combined"
    if [[ -s "$combined/metrics.csv" && -s "$combined/predictions.csv" ]]; then
      echo "SKIP completed $hybrid_label/$split"
      continue
    fi
    for horizon in 1 2 3; do
      local output="$results_root/$hybrid_label/$split/H$horizon"
      mkdir -p "$output"
      "$rscript_bin" "$code_root/r/representation_augmented_inla.R" \
        "$rows_cache" "$adjacency_path" "$representation" "$output" \
        "$horizon" "$draws" 1234 "$hybrid_label" \
        latent_residual_pca 4 5 1 "$split"
    done
    "$python_bin" "$code_root/scripts/combine_augmented_inla_horizons.py" \
      "$combined" \
      "$results_root/$hybrid_label/$split/H1" \
      "$results_root/$hybrid_label/$split/H2" \
      "$results_root/$hybrid_label/$split/H3"
  done
}

run_hybrid DEEP-LN-GNN HYBRID-LN-INLA
run_hybrid DEEP-NB-GNN HYBRID-NB-INLA
run_hybrid DEEP-LAND-GNN HYBRID-LAND-INLA

best_hybrid="$($python_bin - "$results_root" <<'PY'
from pathlib import Path
import sys
import pandas as pd
root = Path(sys.argv[1])
labels = ("HYBRID-LN-INLA", "HYBRID-NB-INLA", "HYBRID-LAND-INLA")
scores = {}
for label in labels:
    metrics = pd.read_csv(root / label / "validation/combined/metrics.csv")
    scores[label] = float(metrics["crps"].mean())
print(min(scores, key=scores.get))
PY
)"
echo "$best_hybrid" > "$results_root/selected_hybrid.txt"

proposal_dir="$results_root/PROPOSAL-PARTIAL-PEAK"
if [[ ! -s "$proposal_dir/metrics.csv" ]]; then
  set +e
  "$python_bin" "$code_root/scripts/tune_partial_peak_gate.py" \
    "$results_root/$best_hybrid/validation/combined" \
    "$results_root/$best_hybrid/test/combined" \
    "$results_root/DEEP-PEAK-GNN" \
    "$results_root/INLA-ONLY-validation/INLA-TS/metrics.csv" \
    "$proposal_dir" --draws "$draws"
  proposal_status=$?
  set -e
  if [[ $proposal_status -ne 0 ]]; then
    echo "Partial peak gate was not validation-feasible" > "$results_root/proposal_status.txt"
  fi
fi

echo "Core 2021-2025 rerun completed at $results_root"
