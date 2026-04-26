#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT="${1:-outputs/gsm8k_deff_scores_k8_t448_tau0.2_20260412_212149.jsonl}"
OUTPUT="${2:-}"
SEED="${3:-42}"

echo "==== Sort DEFF Scores To JSON Start ===="
echo "Python: ${PYTHON_BIN}"
echo "Input: ${INPUT}"
if [ -n "${OUTPUT}" ]; then
  echo "Output: ${OUTPUT}"
else
  echo "Output: <input>_deff_sorted.json"
fi
echo "Seed: ${SEED}"

CMD=(
  "${PYTHON_BIN}" sort_deff_scores_to_json.py
  --input "${INPUT}"
  --seed "${SEED}"
)

if [ -n "${OUTPUT}" ]; then
  CMD+=(--output "${OUTPUT}")
fi

"${CMD[@]}"

echo "==== Sort DEFF Scores To JSON Finished ===="
