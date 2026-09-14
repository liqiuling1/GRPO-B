#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"

NUM_GPUS="${NUM_GPUS:-}"
if [ -n "${NUM_GPUS}" ] && [ -z "${NUM_PROCESSES:-}" ]; then
  NUM_PROCESSES="${NUM_GPUS}"
fi
NUM_PROCESSES="${NUM_PROCESSES:-1}"

mkdir -p logs
TIME=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/train_accuracy_gate_${TIME}.log"
PID_FILE="logs/train_accuracy_gate_latest.pid"
LATEST_LOG_LINK="logs/train_accuracy_gate_latest.log"

usage() {
  cat <<'EOF'
用法:
  bash run_train_accuracy_gate.sh [--mode foreground|background] [options] [checkpoint_path]

核心参数:
  --min_correct_rate X       允许更新的最低 16 答正确率，例如 0.375
  --max_correct_rate X       允许更新的最高 16 答正确率，例如 0.625
  --target_update_steps N    需要的有效更新步数
  --max_attempt_steps N      最多尝试多少个 optimizer step；不传默认 target_update_steps*20

常用训练参数:
  --seed N
  --base_model PATH_OR_NAME
  --dataset_path PATH
  --scores_file PATH
  --start_uid1 N
  --end_uid1 N
  --output_dir PATH
  --batch_size N
  --grad_acc N
  --learning_rate X
  --num_generations N
  --max_completion_length N
  --accepted_length_min N          accepted group 内每个回答的最短 token 数；默认 0
  --accepted_length_max N          accepted group 内每个回答的最长 token 数；默认关闭
  --length_reward_weight X          长度 reward 权重；默认 0，设 0 关闭
  --length_reward_good_min N        不扣长度分的下界；默认 140
  --length_reward_good_max N        不扣长度分的上界；默认 280
  --length_reward_short_min N       低于该长度按最大短回答惩罚处理；默认 0
  --length_reward_max_penalty X     极短或接近 max_completion_length 时的最大扣分；默认 1.0
  --rejected_length_only_update 0|1 被 gate 拒绝但长度过短/过长时，只做负向长度更新；默认 0
  --rejected_length_only_advantage_scale X  rejected length-only 的负 advantage 缩放；默认 length_reward_weight
  --max_retry_truncated_count N       16 个回答中最多允许 N 个截断时补采样；默认 6，设 0 关闭
  --max_truncation_retry_rounds N     截断补采样最多轮数；默认 3，设 0 关闭
  --temperature X
  --top_p X
  --beta X
  --epsilon X
  --save_steps N             每 N 个有效更新保存一次 checkpoint
  --init_adapter_path PATH
  --prompt_style short|fewshot

示例:
  bash run_train_accuracy_gate.sh --min_correct_rate 0.375 --max_correct_rate 0.625 --target_update_steps 50
  bash run_train_accuracy_gate.sh --min_correct_rate 0.5 --max_correct_rate 0.5 --target_update_steps 50
  bash run_train_accuracy_gate.sh --target_update_steps 50 ./grpo_accuracy_gate/checkpoint-50
EOF
}

RUN_MODE="foreground"
RESUME_CKPT=""
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
SEED="${SEED:-42}"
DATASET_PATH="${DATASET_PATH:-}"
TRAIN_SCORES_FILE="${TRAIN_SCORES_FILE:-}"
MIN_UID1="${MIN_UID1:-}"
MAX_UID1="${MAX_UID1:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./grpo_qwen25_15b_gsm8k_lora_accuracy_gate}"
TARGET_UPDATE_STEPS="${TARGET_UPDATE_STEPS:-50}"
MAX_ATTEMPT_STEPS="${MAX_ATTEMPT_STEPS:-}"
MIN_CORRECT_RATE="${MIN_CORRECT_RATE:-0.375}"
MAX_CORRECT_RATE="${MAX_CORRECT_RATE:-0.625}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-448}"
ACCEPTED_LENGTH_MIN="${ACCEPTED_LENGTH_MIN:-0}"
ACCEPTED_LENGTH_MAX="${ACCEPTED_LENGTH_MAX:-}"
LENGTH_REWARD_WEIGHT="${LENGTH_REWARD_WEIGHT:-0}"
LENGTH_REWARD_GOOD_MIN="${LENGTH_REWARD_GOOD_MIN:-}"
LENGTH_REWARD_GOOD_MAX="${LENGTH_REWARD_GOOD_MAX:-}"
LENGTH_REWARD_SHORT_MIN="${LENGTH_REWARD_SHORT_MIN:-0}"
LENGTH_REWARD_MAX_PENALTY="${LENGTH_REWARD_MAX_PENALTY:-1.0}"
REJECTED_LENGTH_ONLY_UPDATE="${REJECTED_LENGTH_ONLY_UPDATE:-0}"
REJECTED_LENGTH_ONLY_ADVANTAGE_SCALE="${REJECTED_LENGTH_ONLY_ADVANTAGE_SCALE:-}"
MAX_RETRY_TRUNCATED_COUNT="${MAX_RETRY_TRUNCATED_COUNT:-6}"
MAX_TRUNCATION_RETRY_ROUNDS="${MAX_TRUNCATION_RETRY_ROUNDS:-3}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
BETA="${BETA:-0.0}"
EPSILON="${EPSILON:-0.2}"
SAVE_STEPS="${SAVE_STEPS:-5}"
INIT_ADAPTER_PATH="${INIT_ADAPTER_PATH:-}"
PROMPT_STYLE="${PROMPT_STYLE:-short}"

while [ $# -gt 0 ]; do
  case "$1" in
    --mode)
      RUN_MODE="$2"
      shift 2
      ;;
    --foreground)
      RUN_MODE="foreground"
      shift
      ;;
    --background)
      RUN_MODE="background"
      shift
      ;;
    --base_model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --dataset_path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --train_scores_file|--scores_file)
      TRAIN_SCORES_FILE="$2"
      shift 2
      ;;
    --min_uid1|--start_uid1)
      MIN_UID1="$2"
      shift 2
      ;;
    --max_uid1|--end_uid1)
      MAX_UID1="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --target_update_steps)
      TARGET_UPDATE_STEPS="$2"
      shift 2
      ;;
    --max_attempt_steps)
      MAX_ATTEMPT_STEPS="$2"
      shift 2
      ;;
    --min_correct_rate)
      MIN_CORRECT_RATE="$2"
      shift 2
      ;;
    --max_correct_rate)
      MAX_CORRECT_RATE="$2"
      shift 2
      ;;
    --batch_size)
      PER_DEVICE_TRAIN_BATCH_SIZE="$2"
      shift 2
      ;;
    --grad_acc)
      GRADIENT_ACCUMULATION_STEPS="$2"
      shift 2
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --num_generations)
      NUM_GENERATIONS="$2"
      shift 2
      ;;
    --max_completion_length)
      MAX_COMPLETION_LENGTH="$2"
      shift 2
      ;;
    --accepted_length_min)
      ACCEPTED_LENGTH_MIN="$2"
      shift 2
      ;;
    --accepted_length_max)
      ACCEPTED_LENGTH_MAX="$2"
      shift 2
      ;;
    --length_reward_weight)
      LENGTH_REWARD_WEIGHT="$2"
      shift 2
      ;;
    --length_reward_good_min)
      LENGTH_REWARD_GOOD_MIN="$2"
      shift 2
      ;;
    --length_reward_good_max)
      LENGTH_REWARD_GOOD_MAX="$2"
      shift 2
      ;;
    --length_reward_short_min)
      LENGTH_REWARD_SHORT_MIN="$2"
      shift 2
      ;;
    --length_reward_max_penalty)
      LENGTH_REWARD_MAX_PENALTY="$2"
      shift 2
      ;;
    --rejected_length_only_update)
      REJECTED_LENGTH_ONLY_UPDATE="$2"
      shift 2
      ;;
    --rejected_length_only_advantage_scale)
      REJECTED_LENGTH_ONLY_ADVANTAGE_SCALE="$2"
      shift 2
      ;;
    --max_retry_truncated_count)
      MAX_RETRY_TRUNCATED_COUNT="$2"
      shift 2
      ;;
    --max_truncation_retry_rounds)
      MAX_TRUNCATION_RETRY_ROUNDS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --top_p)
      TOP_P="$2"
      shift 2
      ;;
    --beta)
      BETA="$2"
      shift 2
      ;;
    --epsilon)
      EPSILON="$2"
      shift 2
      ;;
	    --save_steps)
	      SAVE_STEPS="$2"
      shift 2
      ;;
    --init_adapter_path)
      INIT_ADAPTER_PATH="$2"
      shift 2
      ;;
    --prompt_style)
      PROMPT_STYLE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [ -n "${RESUME_CKPT}" ]; then
        echo "Error: unexpected extra argument: $1" >&2
        usage
        exit 1
      fi
      RESUME_CKPT="$1"
      shift
      ;;
  esac
done

if [ "${RUN_MODE}" != "foreground" ] && [ "${RUN_MODE}" != "background" ]; then
  echo "Error: invalid mode '${RUN_MODE}', expected foreground or background" >&2
  exit 1
fi

DATASET_ARG=()
if [ -n "${DATASET_PATH}" ]; then
  DATASET_ARG=(--dataset_path "${DATASET_PATH}")
fi
TRAIN_SCORES_ARG=()
if [ -n "${TRAIN_SCORES_FILE}" ]; then
  TRAIN_SCORES_ARG=(--train_scores_file "${TRAIN_SCORES_FILE}")
fi
MIN_UID1_ARG=()
if [ -n "${MIN_UID1}" ]; then
  MIN_UID1_ARG=(--min_uid1 "${MIN_UID1}")
fi
MAX_UID1_ARG=()
if [ -n "${MAX_UID1}" ]; then
  MAX_UID1_ARG=(--max_uid1 "${MAX_UID1}")
fi
INIT_ADAPTER_ARG=()
if [ -n "${INIT_ADAPTER_PATH}" ]; then
  INIT_ADAPTER_ARG=(--init_adapter_path "${INIT_ADAPTER_PATH}")
fi
RESUME_ARG=()
if [ -n "${RESUME_CKPT}" ]; then
  RESUME_ARG=(--resume_from_checkpoint "${RESUME_CKPT}")
fi
MAX_ATTEMPT_ARG=()
if [ -n "${MAX_ATTEMPT_STEPS}" ]; then
  MAX_ATTEMPT_ARG=(--max_attempt_steps "${MAX_ATTEMPT_STEPS}")
fi
ACCEPTED_LENGTH_MAX_ARG=()
if [ -n "${ACCEPTED_LENGTH_MAX}" ]; then
  ACCEPTED_LENGTH_MAX_ARG=(--accepted_length_max "${ACCEPTED_LENGTH_MAX}")
fi
LENGTH_REWARD_GOOD_MIN_ARG=()
if [ -n "${LENGTH_REWARD_GOOD_MIN}" ]; then
  LENGTH_REWARD_GOOD_MIN_ARG=(--length_reward_good_min "${LENGTH_REWARD_GOOD_MIN}")
fi
LENGTH_REWARD_GOOD_MAX_ARG=()
if [ -n "${LENGTH_REWARD_GOOD_MAX}" ]; then
  LENGTH_REWARD_GOOD_MAX_ARG=(--length_reward_good_max "${LENGTH_REWARD_GOOD_MAX}")
fi
REJECTED_LENGTH_ONLY_ADVANTAGE_SCALE_ARG=()
if [ -n "${REJECTED_LENGTH_ONLY_ADVANTAGE_SCALE}" ]; then
  REJECTED_LENGTH_ONLY_ADVANTAGE_SCALE_ARG=(--rejected_length_only_advantage_scale "${REJECTED_LENGTH_ONLY_ADVANTAGE_SCALE}")
fi

CMD=(
  accelerate launch
  --mixed_precision bf16
  --num_processes "${NUM_PROCESSES}"
  train_grpo_accuracy_gate.py
  --model_name "${BASE_MODEL}"
  --output_dir "${OUTPUT_DIR}"
  --seed "${SEED}"
  --train_split train
  "${DATASET_ARG[@]}"
  --train_samples -1
  --prompt_style "${PROMPT_STYLE}"
  "${TRAIN_SCORES_ARG[@]}"
  "${MIN_UID1_ARG[@]}"
  "${MAX_UID1_ARG[@]}"
  --target_update_steps "${TARGET_UPDATE_STEPS}"
  "${MAX_ATTEMPT_ARG[@]}"
  --min_correct_rate "${MIN_CORRECT_RATE}"
  --max_correct_rate "${MAX_CORRECT_RATE}"
  --accepted_length_min "${ACCEPTED_LENGTH_MIN}"
  "${ACCEPTED_LENGTH_MAX_ARG[@]}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay 0.0
  --warmup_ratio 0.03
  --num_generations "${NUM_GENERATIONS}"
  --max_completion_length "${MAX_COMPLETION_LENGTH}"
  --length_reward_weight "${LENGTH_REWARD_WEIGHT}"
  "${LENGTH_REWARD_GOOD_MIN_ARG[@]}"
  "${LENGTH_REWARD_GOOD_MAX_ARG[@]}"
  --length_reward_short_min "${LENGTH_REWARD_SHORT_MIN}"
  --length_reward_max_penalty "${LENGTH_REWARD_MAX_PENALTY}"
  --rejected_length_only_update "${REJECTED_LENGTH_ONLY_UPDATE}"
  "${REJECTED_LENGTH_ONLY_ADVANTAGE_SCALE_ARG[@]}"
  --max_retry_truncated_count "${MAX_RETRY_TRUNCATED_COUNT}"
  --max_truncation_retry_rounds "${MAX_TRUNCATION_RETRY_ROUNDS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --beta "${BETA}"
  --epsilon "${EPSILON}"
  --logging_steps 1
  --save_steps "${SAVE_STEPS}"
  --save_total_limit 1000
  --lora_r 16
  --lora_alpha 32
  --lora_dropout 0.05
  --use_4bit
  "${INIT_ADAPTER_ARG[@]}"
  "${RESUME_ARG[@]}"
)

print_summary() {
  echo "==== GRPO Accuracy-Gated Training Start ===="
  echo "Time: ${TIME}"
  echo "Log: ${LOG_FILE}"
  echo "Mode: ${RUN_MODE}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "Accelerate num processes: ${NUM_PROCESSES}"
  echo "Base model: ${BASE_MODEL}"
  echo "Seed: ${SEED}"
  echo "Output dir: ${OUTPUT_DIR}"
  echo "Resume checkpoint: ${RESUME_CKPT:-none}"
  echo "Correct rate interval: [${MIN_CORRECT_RATE}, ${MAX_CORRECT_RATE}]"
  echo "Accepted length interval: [${ACCEPTED_LENGTH_MIN}, ${ACCEPTED_LENGTH_MAX:-disabled}]"
  echo "Target accepted update steps: ${TARGET_UPDATE_STEPS}"
  echo "Max attempt steps: ${MAX_ATTEMPT_STEPS:-target_update_steps*20}"
  echo "Per-device batch size: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
  echo "Gradient accumulation steps: ${GRADIENT_ACCUMULATION_STEPS}"
  echo "Learning rate: ${LEARNING_RATE}"
  echo "Num generations: ${NUM_GENERATIONS}"
  echo "Max completion length: ${MAX_COMPLETION_LENGTH}"
  if [ "${LENGTH_REWARD_WEIGHT}" = "0" ] || [ "${LENGTH_REWARD_WEIGHT}" = "0.0" ]; then
    echo "Length reward: disabled"
  else
    echo "Length reward: weight=${LENGTH_REWARD_WEIGHT}, good_min=${LENGTH_REWARD_GOOD_MIN:-140}, good_max=${LENGTH_REWARD_GOOD_MAX:-280}, short_min=${LENGTH_REWARD_SHORT_MIN}, max_penalty=${LENGTH_REWARD_MAX_PENALTY}"
  fi
  echo "Rejected length-only update: ${REJECTED_LENGTH_ONLY_UPDATE}, advantage_scale=${REJECTED_LENGTH_ONLY_ADVANTAGE_SCALE:-length_reward_weight}"
  echo "Truncation gate: reject prompt groups with any length-truncated completion"
  echo "Truncation retry: max truncated completions ${MAX_RETRY_TRUNCATED_COUNT}, max rounds ${MAX_TRUNCATION_RETRY_ROUNDS}"
  echo "Sampling: temperature=${TEMPERATURE}, top_p=${TOP_P}"
  echo "GRPO regularization: beta=${BETA}, epsilon=${EPSILON}"
  echo "Save accepted-update interval: ${SAVE_STEPS}"
}

print_summary

if [ -f "${PID_FILE}" ]; then
  OLD_PID=$(cat "${PID_FILE}" 2>/dev/null || true)
  if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "A training process is already running with PID ${OLD_PID}."
    echo "Log: $(readlink -f "${LATEST_LOG_LINK}" 2>/dev/null || echo "${LATEST_LOG_LINK}")"
    exit 1
  fi
  rm -f "${PID_FILE}"
fi

ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LOG_LINK}"

if [ "${RUN_MODE}" = "foreground" ]; then
  "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
  echo "==== Accuracy-Gated Training Finished ===="
  echo "Output dir: ${OUTPUT_DIR}"
  exit 0
fi

print_summary > "${LOG_FILE}"
nohup "${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!
echo "${TRAIN_PID}" > "${PID_FILE}"

echo "Started background accuracy-gated training."
echo "PID: ${TRAIN_PID}"
echo "Log: ${LOG_FILE}"
echo "Latest log link: ${LATEST_LOG_LINK}"
echo "Follow log with: tail -f ${LATEST_LOG_LINK}"
echo "Stop with: kill ${TRAIN_PID}"
