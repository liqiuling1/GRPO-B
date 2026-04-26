#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-grpo_b}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
SPLIT="${SPLIT:-train}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
K="${K:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-448}"
TAU="${TAU:-0.2}"
EMBEDDER_DEVICE="${EMBEDDER_DEVICE:-cuda:0}"
KEEP_TOP_N="${KEEP_TOP_N:-20}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-$K}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-2}"
USE_4BIT="${USE_4BIT:-1}"

mkdir -p outputs

TIME_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="k${K}_t${MAX_NEW_TOKENS}_tau${TAU}_${TIME_TAG}"
OUT_FILE="outputs/gsm8k_deff_scores_${RUN_TAG}.jsonl"
FILTERED_FILE="outputs/gsm8k_deff_top_${RUN_TAG}.jsonl"
SUMMARY_FILE="outputs/gsm8k_deff_summary_${RUN_TAG}.json"

echo "==== GSM8K DEFF Filter Start ===="
echo "Conda env: ${CONDA_ENV_NAME}"
echo "HF_HOME: ${HF_HOME}"
echo "Base model: ${BASE_MODEL}"
echo "Split: ${SPLIT}"
echo "Max samples: ${MAX_SAMPLES}"
echo "K: ${K}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "Tau: ${TAU}"
echo "Embedder device: ${EMBEDDER_DEVICE}"
echo "Keep top N: ${KEEP_TOP_N}"
echo "Generation batch size: ${GENERATION_BATCH_SIZE}"
echo "Prompt batch size: ${PROMPT_BATCH_SIZE}"
echo "Use 4bit: ${USE_4BIT}"
echo "Scores out: ${OUT_FILE}"
echo "Filtered out: ${FILTERED_FILE}"
echo "Summary out: ${SUMMARY_FILE}"

CMD=(
  conda run --no-capture-output -n "${CONDA_ENV_NAME}"
  python -u gsm8k_deff_filter.py
  --base_model "${BASE_MODEL}"
  --split "${SPLIT}"
  --max_samples "${MAX_SAMPLES}"
  --K "${K}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --tau "${TAU}"
  --out "${OUT_FILE}"
  --filtered_out "${FILTERED_FILE}"
  --summary_out "${SUMMARY_FILE}"
  --keep_top_n "${KEEP_TOP_N}"
  --generation_batch_size "${GENERATION_BATCH_SIZE}"
  --prompt_batch_size "${PROMPT_BATCH_SIZE}"
  --embedder_device "${EMBEDDER_DEVICE}"
)

if [ "${USE_4BIT}" = "1" ]; then
  CMD+=(--use_4bit)
fi

"${CMD[@]}"

echo "==== GSM8K DEFF Filter Finished ===="
