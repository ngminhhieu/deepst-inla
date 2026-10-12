#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
code_root="$repo_root/code"
python_bin="${DENGUE_PYTHON:-python3}"
rscript_bin="${DENGUE_RSCRIPT:-Rscript}"
config_path="${1:-$code_root/config/qc_lognormal.yaml}"
results_root="${2:-$repo_root/results/research_suite}"
draws="${DENGUE_DRAWS:-500}"
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
export R_LIBS_USER="${DENGUE_R_LIBRARY:-$repo_root/.r-library}"
mkdir -p "$results_root" "$code_root/data/processed/feature_suite"

readarray -t paths < <("$python_bin" - "$config_path" <<'PY'
from pathlib import Path
import sys, yaml
p=Path(sys.argv[1]).resolve()
cfg=yaml.safe_load(p.read_text())
root=p.parent.parent
for key in ("monthly_path", "centroids_path"):
    value=Path(cfg["data"][key])
    print(value if value.is_absolute() else (root/value).resolve())
PY
)
monthly_path="${paths[0]}"
centroids_path="${paths[1]}"
enriched_path="$code_root/data/processed/feature_suite/monthly_commune_enriched.csv"

"$python_bin" "$code_root/scripts/build_feature_panel.py" \
  --monthly "$monthly_path" --centroids "$centroids_path" --output "$enriched_path"

run_deep() {
  local label="$1"; shift
  echo "=== $label ==="
  "$python_bin" "$code_root/scripts/run_experiment.py" \
    --config "$config_path" --model GNN --output-name "$label" \
    --results-root "$results_root" --draws "$draws" "$@"
}

# 1) Pure probabilistic deep learning: same parameter -> distribution -> samples flow.
run_deep DEEP-BASE-LN --export-representations --export-validation-predictions
run_deep DEEP-BASE-NB --likelihood negative_binomial

# 2) Features available already in the current QC panel.
run_deep DEEP-EXTENDED-CLIMATE \
  --monthly-path "$enriched_path" \
  --climate-columns t2m_avg rh_avg tp_accum ws_avg ws_min ws_max t2m_min t2m_max rh_min rh_max rain_3m_sum rain_6m_sum temp_3m_mean humidity_3m_mean

# 3) Human-mobility proxy from gravity connectivity. Replace this with observed
# OD flows via build_mobility_pressure.py when mobile/transport data are available.
run_deep DEEP-MOBILITY-GRAVITY \
  --monthly-path "$enriched_path" \
  --history-covariate-columns gravity_import_pressure

# 4) Static demography / urban-density information that is already derivable.
run_deep DEEP-DENSITY \
  --monthly-path "$enriched_path" \
  --history-covariate-columns log_population_density

# 5) Explicit recent epidemic acceleration features (TCN can learn them too;
# this is an ablation to see whether they help early reaction).
run_deep DEEP-EPIDEMIC-GROWTH \
  --monthly-path "$enriched_path" \
  --history-covariate-columns case_growth_1m case_growth_3m

# 6) Joint all-currently-available feature model.
run_deep DEEP-ALL-AVAILABLE \
  --monthly-path "$enriched_path" \
  --climate-columns t2m_avg rh_avg tp_accum ws_avg ws_min ws_max t2m_min t2m_max rh_min rh_max rain_3m_sum rain_6m_sum temp_3m_mean humidity_3m_mean \
  --history-covariate-columns gravity_import_pressure log_population_density case_growth_1m case_growth_3m

# 7) Separate early-warning objective. Count predictions remain probabilistic;
# an auxiliary outbreak head is trained in parallel and used only for alert probability.
run_deep DEEP-EARLY-WARNING \
  --outbreak-loss-weight "${DENGUE_OUTBREAK_LOSS_WEIGHT:-0.25}" \
  --alert-probability-source auxiliary

# 8) True future climate forecasts, if supplied. IMPORTANT: this file must contain
# forecasts available at issue_date, never realised t+1/t+2/t+3 climate.
if [[ -n "${DENGUE_FUTURE_CLIMATE_PATH:-}" ]]; then
  read -r -a future_columns <<<"${DENGUE_FUTURE_CLIMATE_COLUMNS:-t2m_forecast rh_forecast tp_forecast}"
  run_deep DEEP-FUTURE-CLIMATE \
    --future-covariates-path "$DENGUE_FUTURE_CLIMATE_PATH" \
    --future-covariate-columns "${future_columns[@]}"
  run_deep DEEP-FUTURE-CLIMATE-MOBILITY \
    --monthly-path "$enriched_path" \
    --history-covariate-columns gravity_import_pressure \
    --future-covariates-path "$DENGUE_FUTURE_CLIMATE_PATH" \
    --future-covariate-columns "${future_columns[@]}"
else
  echo "SKIP future-climate models: set DENGUE_FUTURE_CLIMATE_PATH and DENGUE_FUTURE_CLIMATE_COLUMNS."
fi

# 9) Frozen deep representation + Bayesian INLA calibration. This directly tests
# whether INLA adds value beyond the same deep encoder. R-INLA is optional.
if [[ "${DENGUE_RUN_INLA:-1}" == "1" ]] && command -v "$rscript_bin" >/dev/null 2>&1; then
  if "$rscript_bin" -e 'quit(status=ifelse(requireNamespace("INLA", quietly=TRUE),0,1))' >/dev/null 2>&1; then
    DENGUE_RSCRIPT="$rscript_bin" bash "$code_root/scripts/run_deep_inla_all.sh" \
      "$results_root/DEEP-BASE-LN/representations.csv" \
      "$results_root/DEEP-INLA" "$draws" 1234 "DEEP-INLA"
  else
    echo "SKIP DEEP-INLA: R-INLA is not installed."
  fi
else
  echo "SKIP DEEP-INLA: DENGUE_RUN_INLA=0 or Rscript unavailable."
fi

"$python_bin" "$code_root/scripts/compare_model_metrics.py" --results-root "$results_root"
mapfile -t prediction_paths < <(find "$results_root" -mindepth 2 -maxdepth 2 -name predictions.csv -type f | sort)
if ((${#prediction_paths[@]})); then
  "$python_bin" "$code_root/scripts/evaluate_warning_lead_time.py" \
    "${prediction_paths[@]}" --output "$results_root/warning_lead_time.csv"
fi
echo "Research suite completed: $results_root/model_comparison.csv"
