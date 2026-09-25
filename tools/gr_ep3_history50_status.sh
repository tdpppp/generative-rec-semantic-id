#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="$ROOT_DIR/outputs/gr/ep3_history50_pipeline"

if [[ -f "$RUN_ROOT/state.json" ]]; then
    "$ROOT_DIR/.venv/bin/python" -m json.tool "$RUN_ROOT/state.json"
else
    echo "No pipeline state exists yet."
fi

echo
echo "Active GR processes:"
ps -eo pid,etime,%cpu,%mem,stat,cmd | rg 'gr\.(train_gr|evaluate_gr)' | rg -v 'rg ' || true

echo
echo "tmux sessions:"
tmux list-sessions 2>/dev/null || true

echo
echo "GPU:"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits
