#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
code_root="$repo_root/code"
python_bin="${DENGUE_PYTHON:-python3}"
rscript_bin="${DENGUE_RSCRIPT:-/usr/bin/Rscript}"
config_path="${DENGUE_CONFIG:-$code_root/config/eval_2021_2025.yaml}"
results_root="${DENGUE_RESULTS_ROOT:-$repo_root/results/feature_tcn_gnn_2021_2025_v1}"
reference_root="${DENGUE_REFERENCE_ROOT:-$repo_root/results/eval_2021_2025_v1}"
monthly_path="$code_root/data/processed/feature_suite/monthly_commune_open.csv"
adjacency_path="$code_root/data/processed/qc_v1/administrative_adjacency.csv"
rows_cache="$reference_root/inla_rows_cache.rds"
draws="${DENGUE_DRAWS:-500}"

export PYTHONPATH="$code_root:$code_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export R_LIBS_USER="${DENGUE_R_LIBRARY:-$repo_root/.r-library}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export DENGUE_TRAIN_START="${DENGUE_TRAIN_START:-2010-01-01}"
export DENGUE_TRAIN_END="${DENGUE_TRAIN_END:-2018-12-01}"
export DENGUE_VALIDATION_START="${DENGUE_VALIDATION_START:-2019-01-01}"
export DENGUE_VALIDATION_END="${DENGUE_VALIDATION_END:-2020-12-01}"
export DENGUE_TEST_START="${DENGUE_TEST_START:-2021-01-01}"
export DENGUE_TEST_END="${DENGUE_TEST_END:-2025-12-01}"
export DENGUE_SPLIT_ON_TARGET_DATE="${DENGUE_SPLIT_ON_TARGET_DATE:-true}"

selected_feature="$(tr -d '\r\n' < "$results_root/selected_feature_model.txt")"
base=(t2m_avg rh_avg tp_accum)
case "$selected_feature" in
  TCN-GNN-BASE) features=("${base[@]}") ;;
  TCN-GNN-SOIL) features=("${base[@]}" soil_moisture) ;;
  TCN-GNN-SOLAR) features=("${base[@]}" solar_downward) ;;
  TCN-GNN-DEWPOINT) features=("${base[@]}" dewpoint_c) ;;
  TCN-GNN-ET0) features=("${base[@]}" et0_mm) ;;
  TCN-GNN-ALL) features=("${base[@]}" soil_moisture solar_downward dewpoint_c et0_mm) ;;
  *) echo "Unknown selected feature model: $selected_feature" >&2; exit 2 ;;
esac
printf '%s\n' "${features[@]}" > "$results_root/selected_features.txt"

run_architecture() {
  local label="$1"
  local architecture="$2"
  local output="$results_root/$label"
  if [[ ! -s "$output/metrics.csv" || ! -s "$output/validation_predictions.csv" ]]; then
    "$python_bin" "$code_root/scripts/run_experiment.py" \
      --config "$config_path" --model "$architecture" --output-name "$label" \
      --results-root "$results_root" --monthly-path "$monthly_path" \
      --draws "$draws" --export-representations --export-validation-predictions \
      --climate-columns "${features[@]}"
  fi
  if [[ ! -s "$output/validation_metrics.csv" ]]; then
    "$python_bin" "$code_root/scripts/score_saved_validation.py" "$output"
  fi
}

run_architecture ARCH-LAST-GNN GNN
run_architecture ARCH-TCN-GNN-NOGATE M5
run_architecture ARCH-TCN-GNN-GATED M6
run_architecture ARCH-LSTM-GNN LSTM-GNN
run_architecture ARCH-CNN-LSTM-GNN CNN-LSTM-GNN

"$python_bin" - "$results_root" <<'PY'
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
architecture = {
    "ARCH-LAST-GNN": "GNN",
    "ARCH-TCN-GNN-NOGATE": "M5",
    "ARCH-TCN-GNN-GATED": "M6",
    "ARCH-LSTM-GNN": "LSTM-GNN",
    "ARCH-CNN-LSTM-GNN": "CNN-LSTM-GNN",
}
metrics = pd.concat(
    [pd.read_csv(root / label / "validation_metrics.csv") for label in architecture],
    ignore_index=True,
)
metrics.to_csv(root / "architecture_validation_metrics.csv", index=False)
summary = metrics.groupby("model", as_index=False).agg(
    mean_crps=("crps", "mean"), mean_mae=("mae", "mean"),
    mean_brier=("outbreak_brier", "mean"),
    mean_sensitivity=("outbreak_sensitivity", "mean"), mean_f1=("outbreak_f1", "mean"),
).sort_values(["mean_crps", "mean_brier", "mean_mae"])
summary.to_csv(root / "architecture_validation_summary.csv", index=False)
winner = str(summary.iloc[0]["model"])
runner_up = str(summary.iloc[1]["model"])
(root / "selected_architecture_model.txt").write_text(winner + "\n", encoding="utf-8")
(root / "selected_architecture_runner_up.txt").write_text(runner_up + "\n", encoding="utf-8")
(root / "selected_architecture_code.txt").write_text(architecture[winner] + "\n", encoding="utf-8")
print(summary.to_string(index=False))
print(f"Selected by validation mean CRPS: {winner} ({architecture[winner]})")
PY

winner="$(tr -d '\r\n' < "$results_root/selected_architecture_model.txt")"
runner_up="$(tr -d '\r\n' < "$results_root/selected_architecture_runner_up.txt")"
winner_code="$(tr -d '\r\n' < "$results_root/selected_architecture_code.txt")"
peak_dir="$results_root/ARCH-WINNER-PEAK"
if [[ ! -s "$peak_dir/metrics.csv" || ! -s "$peak_dir/validation_predictions.csv" ]]; then
  "$python_bin" "$code_root/scripts/run_experiment.py" \
    --config "$config_path" --model "$winner_code" --output-name ARCH-WINNER-PEAK \
    --results-root "$results_root" --monthly-path "$monthly_path" \
    --draws "$draws" --peak-loss-weight 1 \
    --export-representations --export-validation-predictions \
    --climate-columns "${features[@]}"
fi
if [[ ! -s "$peak_dir/validation_metrics.csv" ]]; then
  "$python_bin" "$code_root/scripts/score_saved_validation.py" "$peak_dir"
fi

deep_stack="$results_root/ARCH-TOP2-STACK"
if [[ ! -s "$deep_stack/metrics.csv" ]]; then
  "$python_bin" "$code_root/scripts/stack_predictive_distributions.py" \
    --output "$deep_stack" --model-label ARCH-TOP2-STACK \
    --members "$results_root/$winner" "$results_root/$runner_up" \
    --selection-draws 100 --crps-tolerance 0.03 --minimum-precision 0.50
fi

for label in ARCH-LAST-GNN ARCH-TCN-GNN-NOGATE ARCH-TCN-GNN-GATED ARCH-LSTM-GNN ARCH-CNN-LSTM-GNN; do
  if [[ "$label" != "$winner" ]]; then
    output="$results_root/$label"
    for artifact in model.pt predictive_samples.npz validation_predictive_samples.npz representations.csv predictions.csv validation_predictions.csv; do
      [[ -e "$output/$artifact" ]] && truncate -s 0 "$output/$artifact"
    done
  fi
done

hybrid_label="ARCH-WINNER-REPR-INLA"
representation="$results_root/$winner/representations.csv"
for split in validation test; do
  combined="$results_root/$hybrid_label/$split/combined"
  if [[ -s "$combined/metrics.csv" && -s "$combined/predictions.csv" ]]; then
    continue
  fi
  for horizon in 1 2 3; do
    output="$results_root/$hybrid_label/$split/H$horizon"
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

ensemble_root="$results_root/ARCH-WINNER-INLA-ENSEMBLES"
if [[ ! -s "$ensemble_root/INLA-GNN-EXACT-SOFT-CRPS/metrics.csv" ]]; then
  "$python_bin" "$code_root/scripts/exact_inla_deep_ensemble.py" \
    "$reference_root/INLA-ONLY-validation/INLA-TS" \
    "$reference_root/INLA-ONLY-test/INLA-TS" \
    "$results_root/$winner" "$ensemble_root" --draws "$draws"
fi

proposal_dir="$results_root/ARCH-WINNER-REPR-INLA-PEAK-GATE"
if [[ ! -s "$proposal_dir/metrics.csv" ]]; then
  set +e
  "$python_bin" "$code_root/scripts/tune_partial_peak_gate.py" \
    "$results_root/$hybrid_label/validation/combined" \
    "$results_root/$hybrid_label/test/combined" \
    "$peak_dir" \
    "$reference_root/INLA-ONLY-validation/INLA-TS/metrics.csv" \
    "$proposal_dir" --draws "$draws"
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    printf 'No validation-feasible partial Peak gate for the selected architecture.\n' \
      > "$results_root/peak_gate_status.txt"
  fi
fi

echo "Architecture and integration ablation completed at $results_root"
