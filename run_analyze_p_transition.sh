#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"

# 手动改这里：第一个文件是训练/变化前，第二个文件是训练/变化后。
BEFORE="${BEFORE:-outputs/merge_final/gsm8k_p_scores_final_no_truncation_keep_final_2_10pct.jsonl}"
AFTER="${AFTER:-outputs/gsm8k_p_scores_final_10%_after_pvar_uid1_0-131/gsm8k_p_scores_final_10%_after_pvar_uid1_0-131_checkpoint-10.jsonl}"

# 手动改这里：你想观察的难度值，比如 0.5。
# 如果想观察区间，设置 OBSERVE_P_RANGE="0.375,0.65625"。
# 区间默认是左闭右开 [left, right)，设置 OBSERVE_P_RANGE_RIGHT_CLOSED=1 后变成 [left, right]。
OBSERVE_P="${OBSERVE_P:-0.5}"
OBSERVE_P_RANGE="${OBSERVE_P_RANGE:-}"
OBSERVE_P_RANGE_RIGHT_CLOSED="${OBSERVE_P_RANGE_RIGHT_CLOSED:-0}"

if [ -n "${OBSERVE_P_RANGE}" ]; then
  OBSERVE_LABEL="${OBSERVE_P_RANGE//,/_to_}"
  if [ "${OBSERVE_P_RANGE_RIGHT_CLOSED}" = "1" ]; then
    OBSERVE_LABEL="${OBSERVE_LABEL}_closed"
  else
    OBSERVE_LABEL="${OBSERVE_LABEL}_open"
  fi
else
  OBSERVE_LABEL="${OBSERVE_P}"
fi

OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs/p_transition_observe_${OBSERVE_LABEL}}"
WRITE_UID_LISTS="${WRITE_UID_LISTS:-0}"

echo "==== Analyze P Transition Start ===="
echo "Python: ${PYTHON_BIN}"
echo "Before: ${BEFORE}"
echo "After: ${AFTER}"
if [ -n "${OBSERVE_P_RANGE}" ]; then
  echo "Observe p range: ${OBSERVE_P_RANGE}"
  echo "Observe p range right closed: ${OBSERVE_P_RANGE_RIGHT_CLOSED}"
else
  echo "Observe p: ${OBSERVE_P}"
fi
echo "Output prefix: ${OUTPUT_PREFIX}"

CMD=(
  "${PYTHON_CMD[@]}" analyze_p_transition.py
  --before "${BEFORE}"
  --after "${AFTER}"
  --output_prefix "${OUTPUT_PREFIX}"
)

if [ -n "${OBSERVE_P_RANGE}" ]; then
  CMD+=(--observe_p_range "${OBSERVE_P_RANGE}")
  if [ "${OBSERVE_P_RANGE_RIGHT_CLOSED}" = "1" ]; then
    CMD+=(--observe_p_range_right_closed)
  fi
else
  CMD+=(--observe_p "${OBSERVE_P}")
fi

if [ "${WRITE_UID_LISTS}" = "1" ]; then
  CMD+=(--write_uid_lists)
fi

"${CMD[@]}"

echo "==== Analyze P Transition Finished ===="
