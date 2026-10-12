#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
session_name="${DENGUE_OPEN_SESSION:-dengue_open_suite}"
python_bin="${DENGUE_PYTHON:-python3}"
rscript_bin="${DENGUE_RSCRIPT:-Rscript}"
log_file="${DENGUE_RUN_LOG:-$repo_root/logs/open_data_suite.log}"
gpu="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$repo_root/logs"
if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "tmux session already exists: $session_name" >&2
  echo "Attach with: tmux attach -t $session_name" >&2
  exit 1
fi

tmux new-session -d -s "$session_name" \
  "cd '$repo_root' && \
    export DENGUE_PYTHON='$python_bin' \
           DENGUE_RSCRIPT='$rscript_bin' \
           R_LIBS_USER='$repo_root/.r-library' \
           CUDA_VISIBLE_DEVICES='$gpu' \
           MPLBACKEND=Agg \
           PYTHONUNBUFFERED=1 \
           OMP_NUM_THREADS=4 \
           MKL_NUM_THREADS=4 \
           PYTHONPATH='$repo_root/code' && \
    '$python_bin' '$repo_root/code/scripts/run_open_data_suite.py' \
      2>&1 | tee -a '$log_file'; \
    echo EXIT_CODE=\$? | tee -a '$log_file'; \
    exec bash"

echo "Started durable tmux session: $session_name"
echo "Log: $log_file"
echo "Status: $repo_root/logs/open_data_suite_status.txt"
echo "Monitor: tmux attach -t $session_name"
echo "Tail: tail -f $log_file"
