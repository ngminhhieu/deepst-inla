#!/usr/bin/env bash
set -euo pipefail

if test "$#" -lt 2; then
  echo "Usage: run_deep_inla_all.sh representations.csv output_dir [draws] [seed] [model_label]" >&2
  exit 2
fi

representations_path="$1"
output_dir="$2"
draws="${3:-500}"
seed="${4:-1234}"
model_label="${5:-M6-INLA}"
rscript_bin="${DENGUE_RSCRIPT:-Rscript}"
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
temporary_root="$(mktemp -d)"
trap 'rm -rf "$temporary_root"' EXIT

for horizon in 1 2 3; do
  horizon_dir="$temporary_root/H$horizon"
  mkdir -p "$horizon_dir"
  "$rscript_bin" "$script_root/../r/m6_inla.R" \
    "$representations_path" "$horizon_dir/predictions.csv" \
    "$draws" "$seed" "$horizon" "$model_label"
done

mkdir -p "$output_dir"
python3 - "$temporary_root" "$output_dir" <<'PY'
from pathlib import Path
import sys
import pandas as pd

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
for filename in ("predictions.csv", "metrics.csv", "diagnostics.csv"):
    frames = [pd.read_csv(source / f"H{horizon}" / filename) for horizon in (1, 2, 3)]
    pd.concat(frames, ignore_index=True).to_csv(destination / filename, index=False)
PY
