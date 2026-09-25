#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="gr_ep3_history50_daemon"
RUN_ROOT="$ROOT_DIR/outputs/gr/ep3_history50_pipeline"
DAEMON_LOG="$RUN_ROOT/daemon.log"

mkdir -p "$RUN_ROOT"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION"
    exit 0
fi

rm -f "$RUN_ROOT/STOP"
tmux new-session -d -s "$SESSION" \
    "cd '$ROOT_DIR' && bash -lc '
        attempts=0
        while (( attempts < 3 )); do
            .venv/bin/python -u tools/run_gr_ep3_history50_pipeline.py
            rc=\$?
            if (( rc != 75 )); then
                exit \$rc
            fi
            attempts=\$((attempts + 1))
            echo \"[\$(date --iso-8601=seconds)] restarting pipeline after internal failure (\$attempts/3)\"
            sleep 30
        done
        exit 75
    ' 2>&1 | tee -a '$DAEMON_LOG'"

echo "started tmux session: $SESSION"
echo "state: $RUN_ROOT/state.json"
echo "log: $DAEMON_LOG"
