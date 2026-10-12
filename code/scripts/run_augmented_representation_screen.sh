#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export R_LIBS_USER="${DENGUE_R_LIBRARY:-$repo_root/.r-library}"
rows_cache="${DENGUE_ROWS_CACHE:-$repo_root/results/inla_rows_cache.rds}"
adjacency="${DENGUE_ADJACENCY_PATH:-$repo_root/code/data/processed/qc_v1/administrative_adjacency.csv}"
output_root="${DENGUE_AUGMENTED_ROOT:-$repo_root/results/representation_augmented_screen}"
draws="${DENGUE_SCREEN_DRAWS:-100}"
parallel="${DENGUE_SCREEN_PARALLEL:-3}"
threads="${DENGUE_SCREEN_THREADS:-4}"

run_one() {
  IFS='|' read -r source input mode components precision label <<<"$1"
  output="$output_root/$label"
  mkdir -p "$output"
  if [[ -s "$output/metrics.csv" ]]; then
    return
  fi
  OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" Rscript \
    "$repo_root/code/r/representation_augmented_inla.R" \
    "$rows_cache" "$adjacency" "$repo_root/$input" "$output" \
    3 "$draws" 1234 "$label" "$mode" "$components" 5 "$precision" validation \
    >"$output/run.log" 2>&1
}
export -f run_one
export repo_root rows_cache adjacency output_root draws threads

jobs=()
for source in REPR-GNN REPR-TCN REPR-TCN-GNN REPR-M6; do
  score_path="results/deep_score_screen/${source}-location_scale.csv"
  latent_path="results/${source}/representations.csv"
  for mode in score_linear score_hinge score_rw2 offset offset_shrink; do
    jobs+=("$source|$score_path|$mode|4|4|AUG-${source}-${mode}")
  done
  for mode in latent_pca latent_residual_pca; do
    for components in 4 12; do
      jobs+=("$source|$latent_path|$mode|$components|4|AUG-${source}-${mode}-P${components}")
    done
  done
done

printf '%s\n' "${jobs[@]}" | xargs -P "$parallel" -I '{}' bash -c 'run_one "$1"' _ '{}'

python_bin="${DENGUE_PYTHON:-python3}"
"$python_bin" - "$output_root" <<'PY'
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
frames = [pd.read_csv(path) for path in root.glob("*/metrics.csv")]
summary = pd.concat(frames, ignore_index=True).sort_values(["crps", "outbreak_f1"])
summary.to_csv(root / "validation_h3_summary.csv", index=False)
print(summary[["model", "crps", "rmse", "outbreak_f1", "outbreak_recall"]].to_string(index=False))
PY
