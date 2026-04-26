#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Prefer the user's complete global Hugging Face cache by default.
# Override HF_HOME explicitly if you want to force a project-local cache.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p logs
TIME=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/smoke_test_${TIME}.log"
: "${LOG_FILE:?LOG_FILE is empty}"

echo "==== GRPO Smoke Test Start ===="
echo "Time: ${TIME}"
echo "Log: ${LOG_FILE}"
echo "HF_HOME: ${HF_HOME}"

python train_grpo.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir ./grpo_qwen25_15b_gsm8k_lora_grpo_smoke_test \
  --seed 42 \
  --train_split train \
  --train_samples 8 \
  --max_steps 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --num_generations 4 \
  --max_completion_length 96 \
  --temperature 1.0 \
  --top_p 0.95 \
  --beta 0.0 \
  --epsilon 0.2 \
  --logging_steps 1 \
  --save_steps 5 \
  --save_total_limit 1 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --use_4bit 2>&1 | tee -a "${LOG_FILE}"

echo "==== Smoke Test Finished ===="
echo "Output dir: ./grpo_qwen25_15b_gsm8k_lora_grpo_smoke_test"
