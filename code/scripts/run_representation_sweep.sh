#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
code_root="$repo_root/code"
python_bin="${DENGUE_PYTHON:-python3}"
config_path="${DENGUE_CONFIG:-$code_root/config/local_lognormal.yaml}"
inla_library="${DENGUE_R_LIBRARY:-$repo_root/.r-library}"

export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
export R_LIBS_USER="$inla_library"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

run_variant() {
  local architecture="$1"
  local source_name="$2"
  local hybrid_name="$3"

  "$python_bin" "$code_root/scripts/run_experiment.py" \
    --config "$config_path" \
    --model "$architecture" \
    --output-name "$source_name" \
    --export-representations

  bash "$code_root/scripts/run_deep_inla_all.sh" \
    "$repo_root/results/$source_name/representations.csv" \
    "$repo_root/results/$hybrid_name" \
    500 1234 "$hybrid_name"
}

# Full-feature temporal representation without graph message passing.
run_variant M3 REPR-TCN TCN-INLA

# Current-state graph representation with four-head attention and a dynamic gate.
run_variant GNN REPR-GNN GNN-INLA

# TCN plus four-head graph attention, without the M6 dynamic gate.
run_variant M5 REPR-TCN-GNN TCN-GNN-INLA
