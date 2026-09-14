#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_INPUT_DIR="outputs/observation_sets_gsm8k_p_intervals_seed42/difficulty_drift_eval_pdist20_seed42"
if [ ! -d "${DEFAULT_INPUT_DIR}" ]; then
  DEFAULT_INPUT_DIR="outputs/difficulty_drift_eval_pdist20_seed42"
fi
INPUT_DIR="${INPUT_DIR:-${DEFAULT_INPUT_DIR}}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${INPUT_DIR}/drift_analysis/difficulty_checkpoint_bin_table}"

"${PYTHON_BIN}" make_difficulty_checkpoint_bin_table.py \
  --input_dir "${INPUT_DIR}" \
  --output_prefix "${OUTPUT_PREFIX}" \
  --include_total \
  "$@"
