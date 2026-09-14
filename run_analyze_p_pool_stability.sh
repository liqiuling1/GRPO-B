#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"

# First file is the original/before pool, second file is the later/after pool.
BEFORE="${BEFORE:-outputs/merge_final/gsm8k_p_scores_final_no_truncation_2.jsonl}"
AFTER="${AFTER:-outputs/merge_final/gsm8k_p_scores_final_no_truncation_2_checkpoint-10.jsonl}"

# Target difficulty interval. Default is [left, right).
P_RANGE="${P_RANGE:-0.375,0.65625}"
P_RANGE_RIGHT_CLOSED="${P_RANGE_RIGHT_CLOSED:-0}"

P_RANGE_LABEL="${P_RANGE//,/_to_}"
if [ "${P_RANGE_RIGHT_CLOSED}" = "1" ]; then
  P_RANGE_LABEL="${P_RANGE_LABEL}_closed"
else
  P_RANGE_LABEL="${P_RANGE_LABEL}_open"
fi

OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs/observe/p_pool_stability_${P_RANGE_LABEL}}"
WRITE_UID_LISTS="${WRITE_UID_LISTS:-0}"

echo "==== Analyze P Pool Stability Start ===="
echo "Python: ${PYTHON_BIN}"
echo "Before: ${BEFORE}"
echo "After: ${AFTER}"
echo "P range: ${P_RANGE}"
echo "P range right closed: ${P_RANGE_RIGHT_CLOSED}"
echo "Output prefix: ${OUTPUT_PREFIX}"

CMD=(
  "${PYTHON_CMD[@]}" analyze_p_pool_stability.py
  --before "${BEFORE}"
  --after "${AFTER}"
  --p_range "${P_RANGE}"
  --output_prefix "${OUTPUT_PREFIX}"
)

if [ "${P_RANGE_RIGHT_CLOSED}" = "1" ]; then
  CMD+=(--p_range_right_closed)
fi

if [ "${WRITE_UID_LISTS}" = "1" ]; then
  CMD+=(--write_uid_lists)
fi

"${CMD[@]}"

echo "==== Analyze P Pool Stability Finished ===="
