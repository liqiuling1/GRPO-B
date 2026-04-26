#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

mkdir -p logs
TIME="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/speed_test_original_gsm8k_${TIME}.log"

OUTPUT_DIR="${OUTPUT_DIR:-./grpo_qwen25_15b_gsm8k_lora_speed_test_original_gsm8k}"
MAX_STEPS="${MAX_STEPS:-20}"
PROMPT_STYLE="${PROMPT_STYLE:-short}"

echo "==== GRPO Speed Test Original GSM8K Start ===="
echo "Log: ${LOG_FILE}"
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
  --save_total_limit 1 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --use_4bit \
  2>&1 | tee "${LOG_FILE}"

echo "==== GRPO Speed Test Original GSM8K Finished ===="
