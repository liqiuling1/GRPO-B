#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
PROMPT_STYLE="${PROMPT_STYLE:-short}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
SEED="${SEED:-42}"
DATASET_PATH="${DATASET_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./grpo_qwen25_15b_gsm8k_lora_grpo_original_safe_lenreward}"
MAX_STEPS="${MAX_STEPS:-7473}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
LENGTH_REWARD_WEIGHT="${LENGTH_REWARD_WEIGHT:-0.5}"
LENGTH_REWARD_GOOD_MIN="${LENGTH_REWARD_GOOD_MIN:-60}"
LENGTH_REWARD_GOOD_MAX="${LENGTH_REWARD_GOOD_MAX:-650}"
LENGTH_REWARD_SHORT_MIN="${LENGTH_REWARD_SHORT_MIN:-0}"
LENGTH_REWARD_MAX_PENALTY="${LENGTH_REWARD_MAX_PENALTY:-1.0}"
SAFE_LENGTH_MIN="${SAFE_LENGTH_MIN:-0}"
SAFE_LENGTH_MAX="${SAFE_LENGTH_MAX:-650}"
REJECTED_LENGTH_ONLY_UPDATE="${REJECTED_LENGTH_ONLY_UPDATE:-1}"
REJECTED_LENGTH_ONLY_ADVANTAGE_SCALE="${REJECTED_LENGTH_ONLY_ADVANTAGE_SCALE:-0.5}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.85}"
BETA="${BETA:-0.001}"
EPSILON="${EPSILON:-0.2}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-7473}"
INIT_ADAPTER_PATH="${INIT_ADAPTER_PATH:-}"
RESUME_CKPT=""

usage() {
  cat <<'EOF'
用法:
  bash run_train_grpo_original_safe_gsm8k.sh [options] [checkpoint_path]

说明:
  原始 GSM8K 数据顺序的 GRPO baseline，但带长度/截断安全保护。
  不使用 accuracy gate，不使用 scores_file 排序。

常用参数:
  --base_model PATH_OR_NAME
  --seed N
  --dataset_path PATH
  --output_dir PATH
  --max_steps N
  --batch_size N
  --grad_acc N
  --learning_rate X
  --num_generations N
  --max_completion_length N
  --length_reward_weight X
  --length_reward_good_min N
  --length_reward_good_max N
  --length_reward_short_min N
  --length_reward_max_penalty X
  --safe_length_min N
  --safe_length_max N
  --rejected_length_only_update 0|1
  --rejected_length_only_advantage_scale X
  --temperature X
  --top_p X
  --beta X
  --epsilon X
  --save_steps N
  --save_total_limit N
  --init_adapter_path PATH
  --prompt_style short|fewshot
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --base_model)
      BASE_MODEL="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --dataset_path)
      DATASET_PATH="$2"; shift 2 ;;
    --output_dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --max_steps)
      MAX_STEPS="$2"; shift 2 ;;
    --batch_size)
      PER_DEVICE_TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --grad_acc)
      GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --learning_rate)
      LEARNING_RATE="$2"; shift 2 ;;
    --num_generations)
      NUM_GENERATIONS="$2"; shift 2 ;;
    --max_completion_length)
      MAX_COMPLETION_LENGTH="$2"; shift 2 ;;
    --length_reward_weight)
      LENGTH_REWARD_WEIGHT="$2"; shift 2 ;;
    --length_reward_good_min)
      LENGTH_REWARD_GOOD_MIN="$2"; shift 2 ;;
    --length_reward_good_max)
      LENGTH_REWARD_GOOD_MAX="$2"; shift 2 ;;
    --length_reward_short_min)
      LENGTH_REWARD_SHORT_MIN="$2"; shift 2 ;;
    --length_reward_max_penalty)
      LENGTH_REWARD_MAX_PENALTY="$2"; shift 2 ;;
    --safe_length_min)
      SAFE_LENGTH_MIN="$2"; shift 2 ;;
    --safe_length_max)
      SAFE_LENGTH_MAX="$2"; shift 2 ;;
    --rejected_length_only_update)
      REJECTED_LENGTH_ONLY_UPDATE="$2"; shift 2 ;;
    --rejected_length_only_advantage_scale)
      REJECTED_LENGTH_ONLY_ADVANTAGE_SCALE="$2"; shift 2 ;;
    --temperature)
      TEMPERATURE="$2"; shift 2 ;;
    --top_p)
      TOP_P="$2"; shift 2 ;;
    --beta)
      BETA="$2"; shift 2 ;;
    --epsilon)
      EPSILON="$2"; shift 2 ;;
    --save_steps)
      SAVE_STEPS="$2"; shift 2 ;;
    --save_total_limit)
      SAVE_TOTAL_LIMIT="$2"; shift 2 ;;
    --init_adapter_path)
      INIT_ADAPTER_PATH="$2"; shift 2 ;;
    --prompt_style)
      PROMPT_STYLE="$2"; shift 2 ;;
    --train_scores_file|--scores_file)
      echo "Error: this script is the no-sorting baseline; do not pass $1" >&2
      usage
      exit 1 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      if [ -n "${RESUME_CKPT}" ]; then
        echo "Error: unexpected extra argument: $1" >&2
        usage
        exit 1
      fi
      RESUME_CKPT="$1"; shift ;;
  esac
done

DATASET_ARG=()
if [ -n "${DATASET_PATH}" ]; then
  DATASET_ARG=(--dataset_path "${DATASET_PATH}")
fi
INIT_ADAPTER_ARG=()
if [ -n "${INIT_ADAPTER_PATH}" ]; then
  INIT_ADAPTER_ARG=(--init_adapter_path "${INIT_ADAPTER_PATH}")
fi
RESUME_ARG=()
if [ -n "${RESUME_CKPT}" ]; then
  RESUME_ARG=(--resume_from_checkpoint "${RESUME_CKPT}")
fi

TIME=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="train_grpo_original_safe_Qwen-GSM8K_lenreward_${TIME}.log"

echo "==== Safe Original-Order GSM8K GRPO Training Start ===="
echo "Log: ${LOG_FILE}"
echo "Base model: ${BASE_MODEL}"
echo "Seed: ${SEED}"
echo "Scores file: none"
echo "Output dir: ${OUTPUT_DIR}"
echo "Max steps: ${MAX_STEPS}"
echo "Batch size: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "Num generations: ${NUM_GENERATIONS}"
echo "Safe length: [${SAFE_LENGTH_MIN}, ${SAFE_LENGTH_MAX}]"
echo "Rejected length-only update: ${REJECTED_LENGTH_ONLY_UPDATE}"

accelerate launch \
  --mixed_precision bf16 \
  --num_processes "${NUM_PROCESSES}" \
  train_grpo_ordered_safe.py \
  --model_name "${BASE_MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --train_split train \
  "${DATASET_ARG[@]}" \
  --train_samples -1 \
  --prompt_style "${PROMPT_STYLE}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --num_generations "${NUM_GENERATIONS}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --length_reward_weight "${LENGTH_REWARD_WEIGHT}" \
  --length_reward_good_min "${LENGTH_REWARD_GOOD_MIN}" \
  --length_reward_good_max "${LENGTH_REWARD_GOOD_MAX}" \
  --length_reward_short_min "${LENGTH_REWARD_SHORT_MIN}" \
  --length_reward_max_penalty "${LENGTH_REWARD_MAX_PENALTY}" \
  --safe_length_min "${SAFE_LENGTH_MIN}" \
  --safe_length_max "${SAFE_LENGTH_MAX}" \
  --rejected_length_only_update "${REJECTED_LENGTH_ONLY_UPDATE}" \
  --rejected_length_only_advantage_scale "${REJECTED_LENGTH_ONLY_ADVANTAGE_SCALE}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --beta "${BETA}" \
  --epsilon "${EPSILON}" \
  --logging_steps 1 \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --use_4bit \
  --report_to none \
  "${INIT_ADAPTER_ARG[@]}" \
  "${RESUME_ARG[@]}" \
  2>&1 | tee "${LOG_FILE}"
