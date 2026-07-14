#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR=/tmp/matplotlib

/home/wzz/anaconda3/bin/conda run -n sat_pro --no-capture-output \
  python comparison/rerun_ga_from_stored.py \
  random_10_100graphs.json \
  --out rerun_ga/random_10_100graphs_ga_fast.json \
  --bp-fast --resume

/home/wzz/anaconda3/bin/conda run -n sat_pro --no-capture-output \
  python comparison/rerun_ga_from_stored.py \
  structured_12_17_100graphs.json \
  --out rerun_ga/structured_12_17_100graphs_ga_fast.json \
  --bp-fast --resume

/home/wzz/anaconda3/bin/conda run -n sat_pro --no-capture-output \
  python comparison/rerun_ga_from_stored.py \
  random_100_10graphs.json \
  --out rerun_ga/random_100_10graphs_ga_fast.json \
  --bp-fast --resume

/home/wzz/anaconda3/bin/conda run -n sat_pro --no-capture-output \
  python comparison/rerun_ga_from_stored.py \
  structured_151_199_10graphs.json \
  --out rerun_ga/structured_151_199_10graphs_ga_fast.json \
  --bp-fast --resume
