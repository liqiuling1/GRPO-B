#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
STATS_DIR="${STATS_DIR:-outputs/observation_sets_gsm8k_p_intervals_seed42/difficulty_drift_eval_pdist20_seed42/difficulty_drift_statistics}"
ORIGINAL_FILE="${ORIGINAL_FILE:-outputs/observation_sets_gsm8k_p_intervals_seed42/difficulty_drift_eval_pdist20_seed42/gsm8k_p_scores_final_no_truncation_2_sampled_20pct.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${STATS_DIR}/representative_sample_trajectories}"

"${PYTHON_BIN}" plot_representative_sample_trajectories.py \
  --per_sample_csv "${STATS_DIR}/per_sample_drift.csv" \
  --original_file "${ORIGINAL_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
