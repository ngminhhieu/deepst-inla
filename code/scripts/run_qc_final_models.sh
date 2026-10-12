#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
code_root="$repo_root/code"
python_bin="${DENGUE_PYTHON:-python3}"
results_root="${DENGUE_RESULTS_ROOT:-$repo_root/results/qc_v1}"
config_path="${DENGUE_CONFIG:-$code_root/config/qc_lognormal.yaml}"
monthly_path="$repo_root/code/data/processed/qc_v1/monthly_commune.csv"
adjacency_path="$repo_root/code/data/processed/qc_v1/administrative_adjacency.csv"
rows_cache="$results_root/inla_rows_cache.rds"
base_rows_cache="${DENGUE_BASE_ROWS_CACHE:-$repo_root/results/inla_rows_cache.rds}"
draws="${DENGUE_FINAL_DRAWS:-500}"

export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
export R_LIBS_USER="${DENGUE_R_LIBRARY:-$repo_root/.r-library}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
mkdir -p "$results_root" "$repo_root/logs"

"$python_bin" "$code_root/scripts/prepare_qc_data.py"
"$python_bin" "$code_root/scripts/validate_data.py" --config "$config_path"

"$python_bin" "$code_root/scripts/run_experiment.py" \
  --config "$config_path" --model GNN --output-name REPR-GNN \
  --results-root "$results_root" --export-representations

"$python_bin" "$code_root/scripts/run_experiment.py" \
  --config "$config_path" --model GNN --output-name MOE-GNN-HIGH \
  --results-root "$results_root" --peak-loss-weight 1 \
  --export-validation-predictions

if [[ -s "$base_rows_cache" ]]; then
  Rscript "$code_root/r/filter_inla_rows_cache.R" \
    "$base_rows_cache" "$monthly_path" "$rows_cache"
else
  rm -f "$rows_cache"
fi
for split in validation test; do
  output="$results_root/inla_${split}_exact"
  mkdir -p "$output"
  DENGUE_EVAL_SPLIT="$split" \
  DENGUE_STATIC_MODELS=INLA-TS \
  DENGUE_SKIP_STATIC_INLA=false \
  DENGUE_RUN_STRONG_INLA=false \
  DENGUE_SAVE_SAMPLES=true \
  DENGUE_ROWS_CACHE="$rows_cache" \
    Rscript "$code_root/r/inla_baselines.R" \
      "$monthly_path" "$adjacency_path" "$output" "$draws" 1234
done

DENGUE_RESULTS_ROOT="$results_root" \
DENGUE_ROWS_CACHE="$rows_cache" \
DENGUE_ADJACENCY_PATH="$adjacency_path" \
DENGUE_FINAL_DRAWS="$draws" \
DENGUE_SKIP_LEGACY_GATE=true \
  bash "$code_root/scripts/run_augmented_representation_final.sh"

PYTHONPATH="$code_root:$code_root/scripts" "$python_bin" \
  "$code_root/scripts/tune_partial_peak_gate.py" \
  "$results_root/inla_gnn_representation_final/validation/INLA-TS-GNN-RES-P4" \
  "$results_root/inla_gnn_representation_final/test/INLA-TS-GNN-RES-P4" \
  "$results_root/MOE-GNN-HIGH" \
  "$results_root/inla_validation_exact/INLA-TS/metrics.csv" \
  "$results_root/proposal_tuned/PROPOSAL-TUNED" --draws "$draws"

"$python_bin" "$code_root/scripts/summarize_qc_final.py" "$results_root"

echo "QC final models completed at $results_root"
