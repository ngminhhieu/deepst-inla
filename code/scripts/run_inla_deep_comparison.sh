#!/usr/bin/env bash
# Direct ablation: probabilistic deep learning vs INLA alone vs deep representation + INLA.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
code_root="$repo_root/code"
python_bin="${DENGUE_PYTHON:-python3}"
rscript_bin="${DENGUE_RSCRIPT:-Rscript}"
config_path="${1:-$code_root/config/qc_lognormal.yaml}"
results_root="${2:-$repo_root/results/inla_deep_comparison}"
draws="${DENGUE_DRAWS:-500}"
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
export R_LIBS_USER="${DENGUE_R_LIBRARY:-$repo_root/.r-library}"
mkdir -p "$results_root"

readarray -t paths < <("$python_bin" - "$config_path" <<'PY'
from pathlib import Path
import sys, yaml
p=Path(sys.argv[1]).resolve()
cfg=yaml.safe_load(p.read_text())
root=p.parent.parent
for key in ("monthly_path", "adjacency_path"):
    value=Path(cfg["data"][key])
    print(value if value.is_absolute() else (root/value).resolve())
PY
)
monthly_path="${paths[0]}"
adjacency_path="${paths[1]}"

# A. Deep-only: distribution parameters -> predictive distribution -> samples.
"$python_bin" "$code_root/scripts/run_experiment.py" \
  --config "$config_path" --model GNN --output-name DEEP-ONLY \
  --results-root "$results_root" --draws "$draws" --export-representations

if ! command -v "$rscript_bin" >/dev/null 2>&1 || \
   ! "$rscript_bin" -e 'quit(status=ifelse(requireNamespace("INLA", quietly=TRUE),0,1))' >/dev/null 2>&1; then
  echo "Deep-only completed. INLA comparison skipped because R-INLA is unavailable."
  "$python_bin" "$code_root/scripts/compare_model_metrics.py" \
    --results-root "$results_root" --models DEEP-ONLY
  exit 0
fi

rows_cache="$results_root/inla_rows_cache.rds"
inla_root="$results_root/INLA-ONLY"
mkdir -p "$inla_root"

# B. INLA-only, fitted/evaluated on exactly the same train/validation/test dates.
DENGUE_EVAL_SPLIT=test \
DENGUE_STATIC_MODELS=INLA-TS \
DENGUE_SKIP_STATIC_INLA=false \
DENGUE_RUN_STRONG_INLA=false \
DENGUE_SAVE_SAMPLES=true \
DENGUE_ROWS_CACHE="$rows_cache" \
  "$rscript_bin" "$code_root/r/inla_baselines.R" \
    "$monthly_path" "$adjacency_path" "$inla_root" "$draws" 1234

# C. Deep + structured INLA. The deep encoder is frozen. INLA then adds its
# structured temporal fixed effects + BYM2 spatial field + the learned z representation.
fusion_root="$results_root/DEEP-INLA"
for horizon in 1 2 3; do
  output="$fusion_root/H$horizon"
  mkdir -p "$output"
  "$rscript_bin" "$code_root/r/representation_augmented_inla.R" \
    "$rows_cache" "$adjacency_path" \
    "$results_root/DEEP-ONLY/representations.csv" "$output" \
    "$horizon" "$draws" 1234 "DEEP-INLA" \
    latent_residual_pca 4 5 1 test
done
"$python_bin" "$code_root/scripts/combine_augmented_inla_horizons.py" \
  "$fusion_root/combined" "$fusion_root/H1" "$fusion_root/H2" "$fusion_root/H3"

"$python_bin" "$code_root/scripts/compare_model_metrics.py" \
  --results-root "$results_root" \
  --models DEEP-ONLY INLA-ONLY/INLA-TS DEEP-INLA/combined \
  --output "$results_root/model_comparison.csv"
"$python_bin" "$code_root/scripts/evaluate_warning_lead_time.py" \
  "$results_root/DEEP-ONLY/predictions.csv" \
  "$results_root/INLA-ONLY/INLA-TS/predictions.csv" \
  "$results_root/DEEP-INLA/combined/predictions.csv" \
  --output "$results_root/warning_lead_time.csv"

echo "Comparison completed: $results_root/model_comparison.csv"
