#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output="$repo_root/results/figure7_multiseed_validation"
mkdir -p "$output"
export PYTHONPATH="$repo_root/code:$repo_root/code/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

python3 "$repo_root/code/paper_pipeline/run_figure7_multiseed.py" architectures \
  --output "$output" 2>&1 | tee -a "$output/architectures.log" &
architecture_pid=$!
python3 "$repo_root/code/paper_pipeline/run_figure7_multiseed.py" features \
  --output "$output" 2>&1 | tee -a "$output/features.log" &
feature_pid=$!

wait "$architecture_pid"
wait "$feature_pid"
python3 "$repo_root/code/paper_pipeline/run_figure7_multiseed.py" summarize \
  --output "$output" 2>&1 | tee -a "$output/summarize.log"
echo "[$(date -Iseconds)] Figure 7 five-seed validation batch complete"
