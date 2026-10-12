#!/usr/bin/env bash
set -euo pipefail

config_path="${1:-config/study_lognormal.yaml}"
python_bin="${DENGUE_PYTHON:-python3}"
monthly_path="${DENGUE_MONTHLY_PATH:?Set DENGUE_MONTHLY_PATH}"
adjacency_path="${DENGUE_ADJACENCY_PATH:?Set DENGUE_ADJACENCY_PATH}"

"$python_bin" scripts/run_baselines.py --config "$config_path"
for model_name in CNN LSTM CNN-LSTM GNN LSTM-FULL LSTM-GNN \
  CNN-LSTM-FULL CNN-LSTM-GNN M1 M2 M3 M4 M5 M6
do
  "$python_bin" scripts/run_experiment.py \
    --config "$config_path" --model "$model_name"
done

"$python_bin" scripts/run_experiment.py \
  --config "$config_path" --model M1 --output-name TCN-H64 --hidden-size 64
"$python_bin" scripts/run_experiment.py \
  --config "$config_path" --model M1 --output-name TCN-L24 \
  --sequence-length 24 --dilations 1 2 4 8

Rscript r/inla_baselines.R \
  "$monthly_path" "$adjacency_path" ../results 500 1234

bash scripts/run_deep_inla_all.sh \
  ../results/M6/representations.csv ../results/M6-INLA 500 1234

"$python_bin" scripts/summarize.py --results-dir ../results
