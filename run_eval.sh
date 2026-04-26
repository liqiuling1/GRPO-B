#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-./grpo_qwen25_15b_gsm8k_lora_grpo_baseline}"
SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
PROMPT_STYLE="${PROMPT_STYLE:-short}"
USE_4BIT="${USE_4BIT:-1}"

echo "==== Evaluation Start ===="
echo "Base model: ${BASE_MODEL}"
echo "Adapter path: ${ADAPTER_PATH}"
echo "Split: ${SPLIT}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "Prompt style: ${PROMPT_STYLE}"
echo "HF_HOME: ${HF_HOME}"
CMD=(
  python evaluate.py
  --base_model "${BASE_MODEL}"
  --adapter_path "${ADAPTER_PATH}"
  --split "${SPLIT}"
  --max_samples "${MAX_SAMPLES}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --prompt_style "${PROMPT_STYLE}"
)

if [ "${USE_4BIT}" = "1" ]; then
  CMD+=(--use_4bit)
fi

"${CMD[@]}"

echo "==== Evaluation Finished ===="
