#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_INPUT_DIR="outputs/observation_sets_gsm8k_p_intervals_seed42/difficulty_drift_eval_pdist20_seed42"
if [ ! -d "${DEFAULT_INPUT_DIR}" ]; then
  DEFAULT_INPUT_DIR="outputs/difficulty_drift_eval_pdist20_seed42"
fi

INPUT_DIR="${INPUT_DIR:-${DEFAULT_INPUT_DIR}}"
ORIGINAL_FILE="${ORIGINAL_FILE:-${INPUT_DIR}/gsm8k_p_scores_final_no_truncation_2_sampled_20pct.jsonl}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${INPUT_DIR}/drift_analysis/difficulty_easy_hard_table}"

"${PYTHON_BIN}" make_difficulty_easy_hard_table.py \
  --input_dir "${INPUT_DIR}" \
  --original_file "${ORIGINAL_FILE}" \
  --output_prefix "${OUTPUT_PREFIX}" \
  "$@"
