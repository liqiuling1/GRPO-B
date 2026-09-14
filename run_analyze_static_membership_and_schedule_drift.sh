#!/bin/bash
set -euo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-grpo_b}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-grpo-b}"
mkdir -p "${MPLCONFIGDIR}"

exec conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
  python -u analyze_static_membership_and_schedule_drift.py "$@"
