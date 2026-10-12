#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runner="$repo_root/code/scripts/run_fixed_report_models_split.sh"

run_one() {
  local tag="$1"
  local config="$2"
  local train_end="$3"
  local validation_start="$4"
  DENGUE_CONFIG="$config" \
  DENGUE_TRAIN_START=2010-01-01 \
  DENGUE_TRAIN_END="$train_end" \
  DENGUE_VALIDATION_START="$validation_start" \
  DENGUE_VALIDATION_END=2020-12-01 \
  DENGUE_TEST_START=2021-01-01 \
  DENGUE_TEST_END=2025-12-01 \
  DENGUE_RESULTS_ROOT="$repo_root/results/model_runs_${tag}_external_thresholds" \
  DENGUE_FINAL_RESULTS_ROOT="$repo_root/results/final_report_${tag}_external_thresholds" \
    bash "$runner"
}

run_one \
  train2010_2018_valid2019_2020_test2021_2025 \
  "$repo_root/code/config/eval_2021_2025.yaml" \
  2018-12-01 2019-01-01

run_one \
  train2010_2015_valid2016_2020_test2021_2025 \
  "$repo_root/code/config/eval_train2010_2015_valid2016_2020_test2021_2025.yaml" \
  2015-12-01 2016-01-01
