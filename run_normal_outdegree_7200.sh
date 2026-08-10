#!/usr/bin/env bash
set -euo pipefail

cd /home/wzz/SAT_pro

PYTHON_BIN=/home/wzz/anaconda3/bin/python
TIMEOUT_SECONDS=7200

echo "[$(date --iso-8601=seconds)] Starting dmax=5 normal out-degree experiments"
"${PYTHON_BIN}" normal_outdegree_experiment.py \
  --timeout "${TIMEOUT_SECONDS}" \
  --output-dir normal_outdegree_dmax5_7200_results \
  --resume \
  --reuse

echo "[$(date --iso-8601=seconds)] Starting dmax=7/10 normal out-degree experiments"
"${PYTHON_BIN}" normal_outdegree_dmax7_10_experiment.py \
  --timeout "${TIMEOUT_SECONDS}" \
  --output-dir normal_outdegree_dmax7_10_7200_results \
  --jobs 1 \
  --resume \
  --reuse

echo "[$(date --iso-8601=seconds)] All normal out-degree experiments completed"
