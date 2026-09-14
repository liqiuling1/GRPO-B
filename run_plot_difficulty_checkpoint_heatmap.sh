#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_CSV="${INPUT_CSV:-outputs/observation_sets_gsm8k_p_intervals_seed42/difficulty_drift_eval_pdist20_seed42/drift_analysis/difficulty_checkpoint_bin_table.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${INPUT_CSV}")}"

if [ ! -f "${INPUT_CSV}" ]; then
  echo "Input CSV not found: ${INPUT_CSV}" >&2
  echo "Run bash run_make_difficulty_checkpoint_bin_table.sh first." >&2
  exit 1
fi

"${PYTHON_BIN}" plot_difficulty_checkpoint_heatmap.py \
  --input_csv "${INPUT_CSV}" \
  --value fraction \
  --output "${OUTPUT_DIR}/difficulty_checkpoint_fraction_heatmap.svg" \
  --title "Difficulty-bin Fraction by Checkpoint" \
  --palette orangered \
  --annotate \
  "$@"

"${PYTHON_BIN}" plot_difficulty_checkpoint_heatmap.py \
  --input_csv "${INPUT_CSV}" \
  --value count \
  --output "${OUTPUT_DIR}/difficulty_checkpoint_count_heatmap.svg" \
  --title "Difficulty-bin Count by Checkpoint" \
  --palette orangered \
  --annotate \
  "$@"
