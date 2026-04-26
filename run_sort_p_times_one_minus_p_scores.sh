#!/bin/bash
set -euo pipefail

INPUT_FILE="${1:-outputs/gsm8k_deff_scores_k8_t448_tau0.2_20260412_212149.jsonl}"
OUTPUT_FILE="${2:-}"

CMD=(
  python sort_p_times_one_minus_p_scores.py
  --input "${INPUT_FILE}"
)

if [ -n "${OUTPUT_FILE}" ]; then
  CMD+=(--output "${OUTPUT_FILE}")
fi

echo "Sorting by p*(1-p) descending..."
echo "Input: ${INPUT_FILE}"
if [ -n "${OUTPUT_FILE}" ]; then
  echo "Output: ${OUTPUT_FILE}"
fi

"${CMD[@]}"
