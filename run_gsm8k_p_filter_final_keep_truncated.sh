#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-grpo_b}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
SPLIT="${SPLIT:-train}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
K="${K:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-1}"
PROMPT_STYLE="${PROMPT_STYLE:-short}"
USE_4BIT="${USE_4BIT:-1}"
START_UID="${START_UID:-}"
END_UID="${END_UID:-}"
RESUME="${RESUME:-0}"
UID_FILE="${UID_FILE:-outputs/gsm8k_p_scores_2048_truncated_need_3072_uids.jsonl}"

mkdir -p outputs

TIME_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="final_keep_truncated_k${K}_t${MAX_NEW_TOKENS}_${TIME_TAG}"
OUT_FILE="${OUT_FILE:-outputs/gsm8k_p_scores_2048_final_keep_truncated.jsonl}"
SUMMARY_FILE="${SUMMARY_FILE:-outputs/gsm8k_p_summary_2048_final_keep_truncated.json}"

echo "==== GSM8K P Final Keep-Truncated Start ===="
echo "Conda env: ${CONDA_ENV_NAME}"
echo "HF_HOME: ${HF_HOME}"
echo "Base model: ${BASE_MODEL}"
echo "Adapter path: ${ADAPTER_PATH:-<none>}"
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
  python -u gsm8k_p_filter_final_keep_truncated.py
  --base_model "${BASE_MODEL}"
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

echo "==== GSM8K P Final Keep-Truncated Finished ===="
