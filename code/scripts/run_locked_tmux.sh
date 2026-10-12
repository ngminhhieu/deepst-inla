#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
code_root="$repo_root/code"
results_root="$repo_root/results"
paper_root="$repo_root/paper"
config_path="${DENGUE_CONFIG:-$code_root/config/qc_lognormal.yaml}"
python_bin="${DENGUE_PYTHON:-python3}"
inla_library="${DENGUE_R_LIBRARY:-$repo_root/.r-library}"
run_log="${DENGUE_RUN_LOG:-$repo_root/logs/locked_experiments.log}"
monthly_path="${DENGUE_MONTHLY_PATH:-$code_root/data/processed/qc_v1/monthly_commune.csv}"
boundaries_path="${DENGUE_BOUNDARIES_PATH:-$code_root/data/private/commune_boundaries.geojson}"
names_path="${DENGUE_NAMES_PATH:-$code_root/data/private/commune_names.csv}"
adjacency_path="${DENGUE_ADJACENCY_PATH:-$code_root/data/processed/qc_v1/administrative_adjacency.csv}"

mkdir -p "$repo_root/logs" "$results_root"
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
export R_LIBS_USER="$inla_library"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

exec > >(tee -a "$run_log") 2>&1

stamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

complete_python_model() {
  local model="$1"
  test -s "$results_root/$model/metrics.csv" \
    && test -s "$results_root/$model/predictions.csv" \
    && { test "$model" != "M6" || test -s "$results_root/M6/representations.csv"; }
}

echo "[$(stamp)] START locked experiment suite"
echo "[$(stamp)] repo=$repo_root"
echo "[$(stamp)] python=$python_bin"
echo "[$(stamp)] config=$config_path"
echo "[$(stamp)] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
"$python_bin" - <<'PY'
import torch
print(
    "torch=", torch.__version__,
    "cuda=", torch.cuda.is_available(),
    "device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
)
PY
Rscript -e 'cat("INLA=", as.character(packageVersion("INLA")), "\n")'

echo "[$(stamp)] Validate input"
"$python_bin" "$code_root/scripts/validate_data.py" --config "$config_path"

for baseline in seasonal_naive province_allocation; do
  if complete_python_model "$baseline"; then
    echo "[$(stamp)] SKIP completed $baseline"
  else
    echo "[$(stamp)] RUN baselines"
    "$python_bin" "$code_root/scripts/run_baselines.py" --config "$config_path"
    break
  fi
done

for model in CNN LSTM CNN-LSTM GNN LSTM-FULL LSTM-GNN \
  CNN-LSTM-FULL CNN-LSTM-GNN M1 M2 M3 M4 M5 M6; do
  if complete_python_model "$model"; then
    echo "[$(stamp)] SKIP completed $model"
    continue
  fi
  echo "[$(stamp)] RUN $model"
  "$python_bin" "$code_root/scripts/run_experiment.py" \
    --config "$config_path" \
    --model "$model"
  echo "[$(stamp)] DONE $model"
done

for sensitivity in TCN-H64 TCN-L24; do
  if complete_python_model "$sensitivity"; then
    echo "[$(stamp)] SKIP completed $sensitivity"
  elif test "$sensitivity" = "TCN-H64"; then
    "$python_bin" "$code_root/scripts/run_experiment.py" --config "$config_path" \
      --model M1 --output-name TCN-H64 --hidden-size 64
  else
    "$python_bin" "$code_root/scripts/run_experiment.py" --config "$config_path" \
      --model M1 --output-name TCN-L24 --sequence-length 24 --dilations 1 2 4 8
  fi
done

if test -s "$results_root/INLA-T/metrics.csv" \
  && test -s "$results_root/INLA-S/metrics.csv" \
  && test -s "$results_root/INLA-TS/metrics.csv"; then
  echo "[$(stamp)] SKIP completed INLA-only baselines"
else
  Rscript "$code_root/r/inla_baselines.R" \
    "$monthly_path" "$adjacency_path" "$results_root" 500 1234
fi

if test -s "$results_root/M6-INLA/metrics.csv" \
  && test "$(tail -n +2 "$results_root/M6-INLA/metrics.csv" | wc -l)" -eq 3 \
  && test -s "$results_root/M6-INLA/predictions.csv"; then
  echo "[$(stamp)] SKIP completed M6-INLA"
else
  echo "[$(stamp)] RUN M6-INLA"
  bash "$code_root/scripts/run_deep_inla_all.sh" \
    "$results_root/M6/representations.csv" "$results_root/M6-INLA" 500 1234
  echo "[$(stamp)] DONE M6-INLA"
fi

echo "[$(stamp)] Build comparisons and manuscript artifacts"
"$python_bin" "$code_root/scripts/summarize.py" --results-dir "$results_root"
"$python_bin" "$code_root/scripts/update_paper_results.py" \
  --results-dir "$results_root" --paper-dir "$paper_root"
"$python_bin" "$code_root/scripts/make_paper_figures.py" \
  --results-dir "$results_root" --paper-dir "$paper_root" \
  --monthly-path "$monthly_path" --boundaries-path "$boundaries_path" \
  --adjacency-path "$adjacency_path" --names-path "$names_path"
"$python_bin" "$code_root/scripts/make_reviewer_figures.py" \
  --results-dir "$results_root" --paper-dir "$paper_root" \
  --names-path "$names_path"
make -C "$paper_root"

echo "[$(stamp)] COMPLETE locked experiment suite"
