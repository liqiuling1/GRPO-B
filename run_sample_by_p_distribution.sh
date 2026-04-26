#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT="${INPUT:-${1:-outputs/gsm8k_deff_scores_k8_t448_tau0.2_20260412_212149.jsonl}}"
SAMPLE_PERCENT="${SAMPLE_PERCENT:-${2:-20}}"
SEED="${SEED:-${3:-42}}"
P_BIN_EDGES="${P_BIN_EDGES:-}"
SAMPLED_OUTPUT="${SAMPLED_OUTPUT:-}"
REMAINING_OUTPUT="${REMAINING_OUTPUT:-}"
SAMPLED_SUMMARY="${SAMPLED_SUMMARY:-}"
REMAINING_SUMMARY="${REMAINING_SUMMARY:-}"

echo "==== Sample By P Distribution Start ===="
echo "Python: ${PYTHON_BIN}"
echo "Input: ${INPUT}"
echo "Sample percent: ${SAMPLE_PERCENT}"
echo "Seed: ${SEED}"
echo "P bin edges: ${P_BIN_EDGES:-<exact p values>}"
echo "Sampled output: ${SAMPLED_OUTPUT:-<auto>}"
echo "Remaining output: ${REMAINING_OUTPUT:-<auto>}"
echo "Sampled summary: ${SAMPLED_SUMMARY:-<auto>}"
echo "Remaining summary: ${REMAINING_SUMMARY:-<auto>}"

CMD=(
  "${PYTHON_BIN}" sample_by_p_distribution.py
  --input "${INPUT}"
  --sample_percent "${SAMPLE_PERCENT}"
  --seed "${SEED}"
)

if [ -n "${P_BIN_EDGES}" ]; then
  CMD+=(--p_bin_edges "${P_BIN_EDGES}")
fi

if [ -n "${SAMPLED_OUTPUT}" ]; then
  CMD+=(--sampled_output "${SAMPLED_OUTPUT}")
fi

if [ -n "${REMAINING_OUTPUT}" ]; then
  CMD+=(--remaining_output "${REMAINING_OUTPUT}")
fi

if [ -n "${SAMPLED_SUMMARY}" ]; then
  CMD+=(--sampled_summary "${SAMPLED_SUMMARY}")
fi

if [ -n "${REMAINING_SUMMARY}" ]; then
  CMD+=(--remaining_summary "${REMAINING_SUMMARY}")
fi

"${CMD[@]}"

echo "==== Sample By P Distribution Finished ===="
