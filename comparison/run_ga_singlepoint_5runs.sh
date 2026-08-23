#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SATPRO_PYTHON="${SATPRO_PYTHON:-python}"
RUN_DIR="${SATPRO_GA_RUN_DIR:-${PROJECT_ROOT}/rerun_ga_singlepoint_5runs}"
LOG_FILE="$RUN_DIR/run.log"
PID_FILE="$RUN_DIR/runner.pid"
STATUS_FILE="$RUN_DIR/status.txt"

mkdir -p "$RUN_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"
cd "${PROJECT_ROOT}"
printf '%s\n' "$BASHPID" > "$PID_FILE"
printf 'running %s\n' "$(date -Is)" > "$STATUS_FILE"
exec >> "$LOG_FILE" 2>&1

finish() {
  run_status=$?
  if [[ $run_status -eq 0 ]]; then
    printf 'complete %s\n' "$(date -Is)" > "$STATUS_FILE"
  else
    printf 'failed exit=%s %s\n' "$run_status" "$(date -Is)" > "$STATUS_FILE"
  fi
}
trap finish EXIT

printf 'started %s pid=%s\n' "$(date -Is)" "$BASHPID"

run_dataset() {
  dataset=$1
  output=$2
  "${SATPRO_PYTHON}" "${PROJECT_ROOT}/comparison/rerun_ga_singlepoint_5runs.py" \
    "$dataset" --out "$output" --runs 5 --bp-fast --resume
}

run_dataset \
  "${PROJECT_ROOT}/structured_12_17_100graphs.json" \
  "$RUN_DIR/structured_12_17_100graphs_ga_singlepoint_5runs.json"
run_dataset \
  "${PROJECT_ROOT}/random_10_100graphs.json" \
  "$RUN_DIR/random_10_100graphs_ga_singlepoint_5runs.json"
run_dataset \
  "${PROJECT_ROOT}/structured_151_199_10graphs.json" \
  "$RUN_DIR/structured_151_199_10graphs_ga_singlepoint_5runs.json"
run_dataset \
  "${PROJECT_ROOT}/random_100_10graphs.json" \
  "$RUN_DIR/random_100_10graphs_ga_singlepoint_5runs.json"

printf 'finished %s\n' "$(date -Is)"
