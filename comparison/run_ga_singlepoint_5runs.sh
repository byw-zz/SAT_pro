#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="/home/wzz/SAT_pro/rerun_ga_singlepoint_5runs"
LOG_FILE="$RUN_DIR/run.log"
PID_FILE="$RUN_DIR/runner.pid"
STATUS_FILE="$RUN_DIR/status.txt"

mkdir -p "$RUN_DIR"
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp/xdg-cache
cd /home/wzz/SAT_pro
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
  /home/wzz/anaconda3/bin/conda run -n sat_pro --no-capture-output \
    python /home/wzz/SAT_pro/comparison/rerun_ga_singlepoint_5runs.py \
    "$dataset" --out "$output" --runs 5 --bp-fast --resume
}

run_dataset \
  /home/wzz/SAT_pro/structured_12_17_100graphs.json \
  "$RUN_DIR/structured_12_17_100graphs_ga_singlepoint_5runs.json"
run_dataset \
  /home/wzz/SAT_pro/random_10_100graphs.json \
  "$RUN_DIR/random_10_100graphs_ga_singlepoint_5runs.json"
run_dataset \
  /home/wzz/SAT_pro/structured_151_199_10graphs.json \
  "$RUN_DIR/structured_151_199_10graphs_ga_singlepoint_5runs.json"
run_dataset \
  /home/wzz/SAT_pro/random_100_10graphs.json \
  "$RUN_DIR/random_100_10graphs_ga_singlepoint_5runs.json"

printf 'finished %s\n' "$(date -Is)"
