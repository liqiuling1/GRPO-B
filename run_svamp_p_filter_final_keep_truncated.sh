#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-grpo_b}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
SPLIT="${SPLIT:-train}"
DATASET_PATH="${DATASET_PATH:-data/svamp-train.arrow}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
K="${K:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-16}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-2}"
PROMPT_STYLE="${PROMPT_STYLE:-short}"
USE_4BIT="${USE_4BIT:-1}"
START_UID="${START_UID:-}"
END_UID="${END_UID:-}"
RESUME="${RESUME:-1}"
UID_FILE="${UID_FILE:-outputs/SVAMP/svamp_p_scores_1024_truncated_need_1536_base_0.5b.jsonl}"

if [ ! -f "${DATASET_PATH}" ] && [ "${DATASET_PATH}" = "data/svamp-train.arrow" ] && [ -f "data/svamp-train-grpo.arrow" ]; then
  DATASET_PATH="data/svamp-train-grpo.arrow"
fi

if [ ! -f "${DATASET_PATH}" ]; then
  echo "Error: SVAMP dataset file does not exist: ${DATASET_PATH}" >&2
  echo "Set DATASET_PATH to your svamp-train.arrow path, or prepare data/svamp-train-grpo.arrow first." >&2
  exit 1
fi

mkdir -p outputs/SVAMP

TIME_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="svamp_final_keep_truncated_k${K}_t${MAX_NEW_TOKENS}_${TIME_TAG}"
OUT_FILE="${OUT_FILE:-outputs/SVAMP/svamp_p_scores_1536_keep_truncated_base_0.5b.jsonl}"
SUMMARY_FILE="${SUMMARY_FILE:-outputs/SVAMP/svamp_p_scores_1536_keep_truncated_base_0.5b_summary.json}"

echo "==== SVAMP P Final Keep-Truncated Start ===="
echo "Conda env: ${CONDA_ENV_NAME}"
echo "HF_HOME: ${HF_HOME}"
echo "Base model: ${BASE_MODEL}"
echo "Adapter path: ${ADAPTER_PATH:-<none>}"
echo "Dataset path: ${DATASET_PATH}"
echo "Split: ${SPLIT}"
echo "Max samples: ${MAX_SAMPLES}"
echo "K: ${K}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "Generation batch size: ${GENERATION_BATCH_SIZE}"
echo "Prompt batch size: ${PROMPT_BATCH_SIZE}"
echo "Prompt style: ${PROMPT_STYLE}"
echo "Use 4bit: ${USE_4BIT}"
echo "Start uid: ${START_UID:-<none>}"
echo "End uid: ${END_UID:-<none>}"
echo "Resume: ${RESUME}"
echo "UID file: ${UID_FILE:-<none>}"
echo "Scores out: ${OUT_FILE}"
echo "Summary out: ${SUMMARY_FILE}"
echo "Run tag: ${RUN_TAG}"

CMD=(
  conda run --no-capture-output -n "${CONDA_ENV_NAME}"
  python -u svamp_p_filter_final_keep_truncated.py
  --base_model "${BASE_MODEL}"
  --dataset_path "${DATASET_PATH}"
  --split "${SPLIT}"
  --max_samples "${MAX_SAMPLES}"
  --K "${K}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --out "${OUT_FILE}"
  --summary_out "${SUMMARY_FILE}"
  --generation_batch_size "${GENERATION_BATCH_SIZE}"
  --prompt_batch_size "${PROMPT_BATCH_SIZE}"
  --prompt_style "${PROMPT_STYLE}"
)

if [ -n "${ADAPTER_PATH}" ]; then
  CMD+=(--adapter_path "${ADAPTER_PATH}")
fi

if [ "${USE_4BIT}" = "1" ]; then
  CMD+=(--use_4bit)
fi

if [ -n "${START_UID}" ]; then
  CMD+=(--start_uid "${START_UID}")
fi

if [ -n "${END_UID}" ]; then
  CMD+=(--end_uid "${END_UID}")
fi

if [ "${RESUME}" = "1" ]; then
  CMD+=(--resume)
fi

if [ -n "${UID_FILE}" ]; then
  CMD+=(--uid_file "${UID_FILE}")
fi

"${CMD[@]}"

echo "==== SVAMP P Final Keep-Truncated Finished ===="
