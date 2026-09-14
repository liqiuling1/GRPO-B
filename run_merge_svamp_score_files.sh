#!/bin/bash
set -euo pipefail

SCORE_FILES="${SCORE_FILES:-outputs/SVAMP/svamp_p_scores_1024_notruncated_base_0.5b.jsonl outputs/SVAMP/svamp_p_scores_1536_keep_truncated_base_0.5b.jsonl}"
OUT_FILE="${OUT_FILE:-outputs/SVAMP/svamp_p_scores_final_base_0.5b.jsonl}"
EXPECTED_COUNT="${EXPECTED_COUNT:-700}"

echo "==== Merge SVAMP Score Files ===="
echo "Score files: ${SCORE_FILES}"
echo "Output file: ${OUT_FILE}"
echo "Expected count: ${EXPECTED_COUNT}"

# shellcheck disable=SC2086
python merge_score_files.py \
  --score_files ${SCORE_FILES} \
  --out "${OUT_FILE}" \
  --expected_count "${EXPECTED_COUNT}"

echo "==== Merge SVAMP Finished ===="
