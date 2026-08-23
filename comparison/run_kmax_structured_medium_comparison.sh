#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SATPRO_PYTHON="${SATPRO_PYTHON:-python}"
OUTPUT_DIR="comparison/kmax_structured_medium_results"
RESULT_JSON="${OUTPUT_DIR}/kmax_structured_medium_comparison.json"

cd "${PROJECT_ROOT}"
exec "${SATPRO_PYTHON}" comparison/kmax_structured_medium_comparison.py \
  --resume \
  --output-dir "${OUTPUT_DIR}" \
  --result-json "${RESULT_JSON}" \
  --khouzani-method bigm \
  --khouzani-milp-time-limit 60 \
  "$@"
