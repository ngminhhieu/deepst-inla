#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export R_LIBS_USER="${DENGUE_R_LIBRARY:-$repo_root/.r-library}"
rows_cache="${DENGUE_ROWS_CACHE:-$repo_root/results/inla_rows_cache.rds}"
adjacency="${DENGUE_ADJACENCY_PATH:-$repo_root/code/data/processed/qc_v1/administrative_adjacency.csv}"
input="$repo_root/results/REPR-GNN/representations.csv"
output_root="${DENGUE_AUGMENTED_TUNING_ROOT:-$repo_root/results/representation_augmented_tuning}"
draws="${DENGUE_SCREEN_DRAWS:-100}"

run_one() {
  IFS='|' read -r components clip precision label <<<"$1"
  output="$output_root/$label"
  mkdir -p "$output"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 Rscript \
    "$repo_root/code/r/representation_augmented_inla.R" \
    "$rows_cache" "$adjacency" "$input" "$output" 3 "$draws" 1234 \
    "$label" latent_residual_pca "$components" "$clip" "$precision" validation \
    >"$output/run.log" 2>&1
}
export -f run_one
export repo_root rows_cache adjacency input output_root draws

jobs=(
  "2|5|4|AUG-GNN-RES-P2-C5-R4"
  "6|5|4|AUG-GNN-RES-P6-C5-R4"
  "8|5|4|AUG-GNN-RES-P8-C5-R4"
  "4|5|1|AUG-GNN-RES-P4-C5-R1"
  "4|5|16|AUG-GNN-RES-P4-C5-R16"
  "4|3|4|AUG-GNN-RES-P4-C3-R4"
  "4|8|4|AUG-GNN-RES-P4-C8-R4"
)
printf '%s\n' "${jobs[@]}" | xargs -P 3 -I '{}' bash -c 'run_one "$1"' _ '{}'

python_bin="${DENGUE_PYTHON:-python3}"
"$python_bin" - "$output_root" <<'PY'
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
x = pd.concat([pd.read_csv(path) for path in root.glob("*/metrics.csv")])
x.sort_values(["crps", "outbreak_f1"]).to_csv(root / "validation_h3_summary.csv", index=False)
print(x.sort_values("crps")[["model", "crps", "rmse", "outbreak_f1"]].to_string(index=False))
PY
