#!/bin/bash
set -euo pipefail

SCORES_FILE="${SCORES_FILE:-outputs/gsm8k_p_scores_manual_shell_1_K32.jsonl}"
KEEP_SCORES_OUT="${KEEP_SCORES_OUT:-outputs/gsm8k_p_scores_448_p1_keep.jsonl}"
REFINE_UIDS_OUT="${REFINE_UIDS_OUT:-outputs/gsm8k_p_scores_448_p_lt_1_need_1024_uids.jsonl}"
THRESHOLD="${THRESHOLD:-1.0}"

echo "==== Split Scores For Refinement ===="
echo "Scores file: ${SCORES_FILE}"
echo "Keep scores out: ${KEEP_SCORES_OUT}"
echo "Refine UIDs out: ${REFINE_UIDS_OUT}"
echo "Threshold: ${THRESHOLD}"

python split_p_for_refine.py \
  --scores_file "${SCORES_FILE}" \
  --keep_scores_out "${KEEP_SCORES_OUT}" \
  --refine_uids_out "${REFINE_UIDS_OUT}" \
  --threshold "${THRESHOLD}"

echo "==== Split Finished ===="
