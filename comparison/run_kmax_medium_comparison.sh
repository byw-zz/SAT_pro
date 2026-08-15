#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SATPRO_PYTHON="${SATPRO_PYTHON:-/home/wzz/anaconda3/bin/python}"

cd "${PROJECT_ROOT}"
exec "${SATPRO_PYTHON}" comparison/kmax_medium_comparison.py \
  --resume \
  --khouzani-method bigm \
  --khouzani-milp-time-limit 60 \
  "$@"
