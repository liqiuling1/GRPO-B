#!/bin/bash
set -euo pipefail

SCORE_FILES="${SCORE_FILES:-outputs/gsm8k_p_scores_448_p1_keep.jsonl outputs/gsm8k_p_scores_1024_nontruncated.jsonl}"
OUT_FILE="${OUT_FILE:-outputs/gsm8k_p_scores_final_no_truncation.jsonl}"
EXPECTED_COUNT="${EXPECTED_COUNT:-7473}"

echo "==== Merge Score Files ===="
echo "Score files: ${SCORE_FILES}"
echo "Output file: ${OUT_FILE}"
echo "Expected count: ${EXPECTED_COUNT}"

# shellcheck disable=SC2086
python merge_score_files.py \
  --score_files ${SCORE_FILES} \
  --out "${OUT_FILE}" \
  --expected_count "${EXPECTED_COUNT}"

echo "==== Merge Finished ===="
