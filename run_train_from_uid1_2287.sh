#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

mkdir -p logs
TIME="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/train_from_uid1_2287_${TIME}.log"

TRAIN_SCORES_FILE="${TRAIN_SCORES_FILE:-outputs/gsm8k_deff_scores_k8_t448_tau0.2_20260412_212149_sorted.jsonl}"
MIN_UID1="${MIN_UID1:-2287}"
OUTPUT_DIR="${OUTPUT_DIR:-./grpo_qwen25_15b_gsm8k_lora_from_uid1_2287}"
MAX_STEPS="${MAX_STEPS:-2593}"
PROMPT_STYLE="${PROMPT_STYLE:-short}"

echo "==== GRPO Training From Sorted Scores Start ===="
echo "Log: ${LOG_FILE}"
echo "Train scores file: ${TRAIN_SCORES_FILE}"
echo "Min uid1: ${MIN_UID1}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Max steps: ${MAX_STEPS}"
echo "Prompt style: ${PROMPT_STYLE}"

accelerate launch \
  --mixed_precision bf16 \
  --num_processes 1 \
  train_grpo.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir "${OUTPUT_DIR}" \
  --seed 42 \
  --train_split train \
  --train_samples -1 \
  --prompt_style "${PROMPT_STYLE}" \
  --train_scores_file "${TRAIN_SCORES_FILE}" \
  --min_uid1 "${MIN_UID1}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --num_generations 4 \
  --max_completion_length 256 \
  --temperature 1.0 \
  --top_p 0.95 \
  --beta 0.0 \
  --epsilon 0.2 \
  --logging_steps 1 \
  --save_steps 100 \
  --save_total_limit 1000 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --use_4bit \
  2>&1 | tee "${LOG_FILE}"

echo "==== GRPO Training From Sorted Scores Finished ===="
