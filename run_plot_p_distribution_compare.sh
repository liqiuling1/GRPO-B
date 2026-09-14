#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-conda run -n grpo_b python}"
OUTPUT="${OUTPUT:-plots/gsm8k_p_distribution_compare.png}"
P_FIELD="${P_FIELD:-p}"
TITLE="${TITLE:-GSM8K P Difficulty Distribution}"
SHOW_VALUES="${SHOW_VALUES:-0}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"

# 手动改这里：想对比几个文件，就在 INPUTS 里放几个文件。
INPUTS=(
  "outputs/gsm8k_p_scores_final_10%.jsonl"
  "outputs/gsm8k_p_scores_final_10%_after_pvar_uid1_0-131_checkpoint-10.jsonl"
  "outputs/gsm8k_p_scores_final_10%_after_pvar_uid1_0-131_checkpoint-20.jsonl"
)

# 手动改这里：LABELS 数量要和 INPUTS 一样，用于图例显示。
LABELS=(
  "final_10%"
  "checkpoint-10"
  "checkpoint-20"
)

echo "==== Plot P Distribution Compare Start ===="
echo "Python: ${PYTHON_BIN}"
echo "Output: ${OUTPUT}"
echo "P field: ${P_FIELD}"
echo "Inputs:"
printf '  %s\n' "${INPUTS[@]}"

CMD=(
  "${PYTHON_CMD[@]}" plot_p_distribution_compare.py
  --inputs "${INPUTS[@]}"
  --labels "${LABELS[@]}"
  --output "${OUTPUT}"
  --p_field "${P_FIELD}"
  --title "${TITLE}"
)

if [ "${SHOW_VALUES}" = "1" ]; then
  CMD+=(--show_values)
fi

"${CMD[@]}"

echo "==== Plot P Distribution Compare Finished ===="
