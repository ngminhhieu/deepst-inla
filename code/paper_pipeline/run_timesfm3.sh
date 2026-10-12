#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/code/scripts/run_timesfm3_gnn_comparison.py" "$@"
# After successful run, copy the resulting summary to results/pending/timesfm3_summary.csv if the runner's output schema changes.
