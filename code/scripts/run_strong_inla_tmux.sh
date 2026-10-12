#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
session_name="${DENGUE_STRONG_SESSION:-dengue_strong_inla}"
monthly_path="${DENGUE_MONTHLY_PATH:-$repo_root/code/data/processed/qc_v1/monthly_commune.csv}"
adjacency_path="${DENGUE_ADJACENCY_PATH:-$repo_root/code/data/processed/qc_v1/administrative_adjacency.csv}"
inla_library="${DENGUE_R_LIBRARY:-$repo_root/.r-library}"
models="${DENGUE_STRONG_MODELS:-INLA-T-AR1,INLA-ST-AR1,INLA-ST-DYN}"

if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "tmux session already exists: $session_name" >&2
  exit 1
fi

mkdir -p "$repo_root/logs"
tmux new-session -d -s "$session_name" \
  "cd '$repo_root' && export R_LIBS_USER='$inla_library' DENGUE_RUN_STRONG_INLA=true DENGUE_SKIP_STATIC_INLA=true DENGUE_STRONG_MODELS='$models' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 && Rscript code/r/inla_baselines.R '$monthly_path' '$adjacency_path' results 500 1234 2>&1 | tee logs/strong_inla.log"

echo "Started tmux session: $session_name"
echo "Monitor with: tmux attach -t $session_name"
echo "Log: $repo_root/logs/strong_inla.log"
