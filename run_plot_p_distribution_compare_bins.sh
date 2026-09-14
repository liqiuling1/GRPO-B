#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-conda run -n grpo_b python}"
OUTPUT="${OUTPUT:-plots/gsm8k_p_compare_ckpt30_vs_ckpt50_bins.png}"
P_FIELD="${P_FIELD:-p}"
TITLE="${TITLE:-GSM8K P Difficulty Distribution by Interval}"
SHOW_VALUES="${SHOW_VALUES:-1}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"

# 手动改这里：区间边界。规则是 [左, 右)，最后一个区间是 [左, 右]。
# 例如下面会得到 [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1]。
P_BIN_EDGES="${P_BIN_EDGES:-0,0.25,0.5,0.75,1}"

# 手动改这里：想对比几个文件，就在 INPUTS 里放几个文件。
INPUTS=(
  "outputs/merge_final/gsm8k_p_scores_final_no_truncation_2_checkpoint-30.jsonl"
  "outputs/merge_final/gsm8k_p_scores_final_no_truncation_2_checkpoint-50.jsonl"
)

# 手动改这里：LABELS 数量要和 INPUTS 一样，用于图例显示。
LABELS=(
  "checkpoint-30"
  "checkpoint-50"
)

echo "==== Plot P Distribution Compare by Bins Start ===="
echo "Python: ${PYTHON_BIN}"
echo "Output: ${OUTPUT}"
echo "P field: ${P_FIELD}"
echo "P bin edges: ${P_BIN_EDGES}"
echo "Inputs:"
printf '  %s\n' "${INPUTS[@]}"

CMD=(
  "${PYTHON_CMD[@]}" plot_p_distribution_compare_bins.py
  --inputs "${INPUTS[@]}"
  --labels "${LABELS[@]}"
  --p_bin_edges "${P_BIN_EDGES}"
  --output "${OUTPUT}"
  --p_field "${P_FIELD}"
  --title "${TITLE}"
)

if [ "${SHOW_VALUES}" = "1" ]; then
  CMD+=(--show_values)
fi

"${CMD[@]}"

echo "==== Plot P Distribution Compare by Bins Finished ===="
