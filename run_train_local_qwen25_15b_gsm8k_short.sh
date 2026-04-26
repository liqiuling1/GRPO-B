#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

HF_ROOT="${HF_HOME:-$HOME/.cache/huggingface}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"
DATASET_PATH="${DATASET_PATH:-}"
BASE_MODEL_PATH_SOURCE="auto-detected"
DATASET_PATH_SOURCE="auto-detected"
OUTPUT_DIR="${OUTPUT_DIR:-./grpo_qwen25_15b_gsm8k_lora_grpo_baseline_local_short}"
PROMPT_STYLE="${PROMPT_STYLE:-short}"

resolve_base_model_path() {
  if [ -n "${BASE_MODEL_PATH}" ]; then
    echo "${BASE_MODEL_PATH}"
    return 0
  fi

  local snapshots_dir="${HF_ROOT}/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots"
  if [ ! -d "${snapshots_dir}" ]; then
    return 1
  fi

  find "${snapshots_dir}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

resolve_dataset_path() {
  if [ -n "${DATASET_PATH}" ]; then
    echo "${DATASET_PATH}"
    return 0
  fi

  local candidate=""
  candidate=$(find "${HF_ROOT}/datasets/openai___gsm8k" -type f -name 'gsm8k-train.arrow' 2>/dev/null | sort | tail -n 1 || true)
  if [ -n "${candidate}" ]; then
    echo "${candidate}"
    return 0
  fi

  candidate=$(find "${HF_ROOT}/datasets/gsm8k" -type f -name 'gsm8k-train.arrow' 2>/dev/null | sort | tail -n 1 || true)
  if [ -n "${candidate}" ]; then
    echo "${candidate}"
    return 0
  fi

  return 1
}

mkdir -p logs
TIME=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/train_local_qwen25_15b_gsm8k_short_${TIME}.log"
PID_FILE="logs/train_local_qwen25_15b_gsm8k_short_latest.pid"
LATEST_LOG_LINK="logs/train_local_qwen25_15b_gsm8k_short_latest.log"

usage() {
  cat <<'EOF'
用法:
  bash run_train_local_qwen25_15b_gsm8k_short.sh [--mode foreground|background] [--dataset_path PATH] [checkpoint_path]
  bash run_train_local_qwen25_15b_gsm8k_short.sh [--foreground|--background] [--dataset_path PATH] [checkpoint_path]

说明:
  使用本地基础模型路径和本地 GSM8K train.arrow 训练 GRPO。
  prompt 固定为 short，对应:
  Solve the following grade school math problem.
  Show your reasoning briefly, and end with a final sentence exactly in the form 'The answer is <number>.'

  Question: {question}

示例:
  bash run_train_local_qwen25_15b_gsm8k_short.sh
  bash run_train_local_qwen25_15b_gsm8k_short.sh --dataset_path /path/to/gsm8k-train.arrow
  bash run_train_local_qwen25_15b_gsm8k_short.sh --background
  bash run_train_local_qwen25_15b_gsm8k_short.sh --background ./grpo_qwen25_15b_gsm8k_lora_grpo_baseline_local_short/checkpoint-500
EOF
}

RUN_MODE="foreground"
RESUME_CKPT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --mode)
      if [ $# -lt 2 ]; then
        echo "Error: --mode requires a value: foreground or background"
        usage
        exit 1
      fi
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
    --dataset_path)
      if [ $# -lt 2 ]; then
        echo "Error: --dataset_path requires a value"
        usage
        exit 1
      fi
      DATASET_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [ -n "${RESUME_CKPT}" ]; then
        echo "Error: unexpected extra argument: $1"
        usage
        exit 1
      fi
      RESUME_CKPT="$1"
      shift
      ;;
  esac
done

if [ -n "${BASE_MODEL_PATH}" ]; then
  BASE_MODEL_PATH_SOURCE="user-specified"
fi
if [ -n "${DATASET_PATH}" ]; then
  DATASET_PATH_SOURCE="user-specified"
fi

BASE_MODEL_PATH="$(resolve_base_model_path || true)"
DATASET_PATH="$(resolve_dataset_path || true)"

if [ "${RUN_MODE}" != "foreground" ] && [ "${RUN_MODE}" != "background" ]; then
  echo "Error: invalid mode '${RUN_MODE}', expected 'foreground' or 'background'"
  usage
  exit 1
fi

if [ ! -d "${BASE_MODEL_PATH}" ]; then
  echo "Error: base model path not found." >&2
  echo "Searched under: ${HF_ROOT}/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots" >&2
  echo "You can also set BASE_MODEL_PATH=/absolute/path/to/snapshot" >&2
  exit 1
fi

if [ ! -f "${DATASET_PATH}" ]; then
  echo "Error: GSM8K train dataset path not found." >&2
  echo "Searched under: ${HF_ROOT}/datasets/openai___gsm8k and ${HF_ROOT}/datasets/gsm8k" >&2
  echo "You can also set DATASET_PATH=/absolute/path/to/gsm8k-train.arrow" >&2
  exit 1
fi

if ! command -v accelerate >/dev/null 2>&1; then
  echo "Error: 'accelerate' command not found in the current environment." >&2
  exit 1
fi

ACCELERATE_CMD=(accelerate launch)

RESUME_ARG=()
if [ -n "${RESUME_CKPT}" ]; then
  RESUME_ARG=(--resume_from_checkpoint "${RESUME_CKPT}")
fi

echo "==== TRL GRPO Local GSM8K Training Start ===="
echo "Time: ${TIME}"
echo "Log: ${LOG_FILE}"
echo "HF_HOME: ${HF_HOME}"
echo "Mode: ${RUN_MODE}"
echo "Base model path (${BASE_MODEL_PATH_SOURCE}): ${BASE_MODEL_PATH}"
echo "Dataset path (${DATASET_PATH_SOURCE}): ${DATASET_PATH}"
echo "Accelerate launcher: ${ACCELERATE_CMD[*]}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Prompt style: ${PROMPT_STYLE}"

if [ -n "${RESUME_CKPT}" ]; then
  echo "Resume from checkpoint: ${RESUME_CKPT}"
else
  echo "Training from scratch"
fi

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

CMD=(
  "${ACCELERATE_CMD[@]}"
  --mixed_precision bf16
  --num_processes 1
  train_grpo.py
  --model_name "${BASE_MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --seed 42
  --train_split train
  --dataset_path "${DATASET_PATH}"
  --train_samples -1
  --prompt_style "${PROMPT_STYLE}"
  --max_steps 3737
  --per_device_train_batch_size 8
  --gradient_accumulation_steps 1
  --learning_rate 1e-5
  --weight_decay 0.0
  --warmup_ratio 0.03
  --num_generations 4
  --max_completion_length 256
  --temperature 1.0
  --top_p 0.95
  --beta 0.0
  --epsilon 0.2
  --logging_steps 1
  --save_steps 100
  --save_total_limit 1000
  --lora_r 16
  --lora_alpha 32
  --lora_dropout 0.05
  --use_4bit
  "${RESUME_ARG[@]}"
)

if [ "${RUN_MODE}" = "foreground" ]; then
  "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
  echo "==== Training Finished ===="
  echo "Output dir: ${OUTPUT_DIR}"
  exit 0
fi

{
  echo "==== TRL GRPO Local GSM8K Training Start ===="
  echo "Time: ${TIME}"
  echo "Log: ${LOG_FILE}"
  echo "HF_HOME: ${HF_HOME}"
  echo "Mode: ${RUN_MODE}"
  echo "Base model path (${BASE_MODEL_PATH_SOURCE}): ${BASE_MODEL_PATH}"
  echo "Dataset path (${DATASET_PATH_SOURCE}): ${DATASET_PATH}"
  echo "Accelerate launcher: ${ACCELERATE_CMD[*]}"
  echo "Output dir: ${OUTPUT_DIR}"
  echo "Prompt style: ${PROMPT_STYLE}"
  if [ -n "${RESUME_CKPT}" ]; then
    echo "Resume from checkpoint: ${RESUME_CKPT}"
  else
    echo "Training from scratch"
  fi
} > "${LOG_FILE}"

nohup "${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!
echo "${TRAIN_PID}" > "${PID_FILE}"

echo "Started background training."
echo "PID: ${TRAIN_PID}"
echo "Log: ${LOG_FILE}"
echo "Latest log link: ${LATEST_LOG_LINK}"
echo "Follow log with: tail -f ${LATEST_LOG_LINK}"
echo "Stop with: kill ${TRAIN_PID}"
